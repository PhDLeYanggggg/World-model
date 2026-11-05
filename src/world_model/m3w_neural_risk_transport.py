"""Source-only risk regression and fixed-reference residual transport."""
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.evaluation import m3w_candidate_neighbor_probe as base
from src.world_model import m3w_rollout_history_risk as storage

SETTINGS = dict(seed=17, width1=64, width2=32, epochs=32, batch_size=512,
    learning_rate=.001, weight_decay=.0001, gradient_clip=5., checkpoint_steps=128)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prepare(data, source):
    x, y, w, sites = (np.asarray(data[k]) for k in ('x', 'target', 'weights', 'sites'))
    n = len(x); fit = sites != source
    if (x.shape != (n, 120) or y.shape != (n, 2) or w.shape != (n,)
            or not fit.any() or fit.all() or not all(np.isfinite(v).all() for v in (x,y,w))
            or (w < 0).any() or w[fit].sum() <= 0):
        raise ValueError('Finite original120 features, separate labels, source-disjoint fitting required')
    z, norm = base.normalized(x, fit, w)
    ww = w[fit]/w[fit].sum()
    scale = np.sqrt(np.einsum('n,nc->c', ww, y[fit]**2)).clip(1e-6)
    fit_ids = data['ids'][fit]
    binding = dict(**norm, target_scale=scale, train_ids=fit_ids,
        train_x_hash=digest(x[fit]), train_y_hash=digest(y[fit]), train_weights_hash=digest(ww),
        train_sources=sorted(set(sites[fit])), held_source=str(source))
    return dict(x=torch.from_numpy(z[fit,1:].astype(np.float32)),
        y=torch.from_numpy((y[fit]/scale).astype(np.float32)),
        w=torch.from_numpy((ww*fit.sum()).astype(np.float32)),
        prior=(ww@y[fit]/scale).astype(np.float32), binding=binding)


class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(120,cfg['width1']),nn.SiLU(),
            nn.Linear(cfg['width1'],cfg['width2']),nn.SiLU(),nn.Linear(cfg['width2'],2))

    def forward(self, x):
        return self.net(x)


def fit(data, source, cfg, identity, path, *, resume=False, stop_steps=None,
        heartbeat=lambda **kw: None, guard=lambda: None):
    p = prepare(data,source); path=Path(path)
    if cfg != SETTINGS:
        raise ValueError('Registered fixed neural budget required')
    torch.manual_seed(cfg['seed']); model=Head(cfg)
    nn.init.zeros_(model.net[-1].weight)
    with torch.no_grad(): model.net[-1].bias.copy_(torch.from_numpy(p['prior']))
    initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    n=len(p['x']); step=epoch=cursor=0; order=torch.empty(0,dtype=torch.long)
    visits=np.zeros(n,np.int32); trace=[]; elapsed=0.
    binding=dict(identity=identity,settings=cfg,preprocess=p['binding'],initial_model=initial,
        fit_rows=n,parameters=sum(v.numel() for v in model.parameters()),
        objective='source_query_weighted_original_two_channel_normalized_MSE',
        threshold_search=False,epochs=cfg['epochs'])
    if path.exists():
        if not resume: raise ValueError('Existing neural fit requires explicit resume')
        saved=storage.base.read_checkpoint(path)
        for key,value in binding.items(): storage.base.exact(saved[key],value)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']); torch.set_rng_state(saved['torch_rng'])
        epoch,cursor,order,step,visits,trace,elapsed=(saved[k] for k in
            ('epoch','cursor','order','step','visits','trace','seconds'))
    started=time.monotonic(); maxsteps=cfg['epochs']*((n+cfg['batch_size']-1)//cfg['batch_size'])
    limit=maxsteps if stop_steps is None else stop_steps
    if not step <= limit <= maxsteps: raise ValueError('Fixed nondecreasing step budget required')
    def save():
        path.parent.mkdir(parents=True,exist_ok=True)
        storage.base.save_checkpoint(path,dict(**binding,model=model.state_dict(),optimizer=optimizer.state_dict(),
            generator=generator.get_state(),torch_rng=torch.get_rng_state(),epoch=epoch,cursor=cursor,
            order=order,step=step,visits=visits,trace=trace,seconds=elapsed+time.monotonic()-started,
            complete=step==maxsteps))
    if not path.exists(): save()
    try:
        while step < limit:
            guard()
            if cursor==n or not len(order):
                order=torch.randperm(n,generator=generator); cursor=0
            rows=order[cursor:cursor+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            pred=model(p['x'][rows]); loss=(p['w'][rows,None]*(pred-p['y'][rows])**2).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite neural risk loss')
            loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); step+=1; cursor+=len(rows); visits[rows.numpy()]+=1
            if cursor==n: epoch+=1
            if step==1 or cursor==n:
                with torch.no_grad():
                    total=0.
                    for begin in range(0,n,1024):
                        sl=slice(begin,begin+1024)
                        total+=float((p['w'][sl,None]*(model(p['x'][sl])-p['y'][sl])**2).sum())
                row=dict(step=step,epoch=epoch,weighted_fit_MSE=total/(2*n),
                    batch_loss=float(loss.detach()),gradient_norm=float(norm),minimum_visits=int(visits.min()))
                trace.append(row); heartbeat(state='neural_risk_training',source=source,**row)
            if step%cfg['checkpoint_steps']==0 or cursor==n or step==limit: save()
    except (KeyboardInterrupt,TimeoutError):
        save(); raise
    out=storage.base.read_checkpoint(path)
    if out['complete'] and not (out['visits']==cfg['epochs']).all():
        raise ValueError('Every donor-fitting row must receive all32 epochs')
    return out


def fit_ridge(data, source):
    fit=data['sites']!=source
    _,state=base.fit_predict(data['x'],data['target'],fit,~fit,data['weights'],penalty=.01)
    return dict(kind='ridge',state=state)


def predict(state, x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=120 or not np.isfinite(x).all():
        raise ValueError('Finite causal120 only; no future-label input')
    if state.get('kind')=='ridge':
        s=state['state']; z=np.clip((x-s['mean'])/s['std'],-8,8)
        return np.column_stack((np.ones(len(x)),z))@s['coefficient']*s['target_scale']
    p=state['preprocess']; cfg=state['settings']; model=Head(cfg)
    model.load_state_dict(state['model']); model.eval()
    z=np.clip((x-p['mean'])/p['std'],-8,8).astype(np.float32); result=[]
    with torch.no_grad():
        for i in range(0,len(x),256): result.append(model(torch.from_numpy(z[i:i+256])).numpy())
    return np.concatenate(result).astype(float)*p['target_scale']


def transport(query_prediction, donor_prediction, donor_labels, weights):
    q,d,y,w=map(np.asarray,(query_prediction,donor_prediction,donor_labels,weights))
    if (q.ndim!=2 or q.shape[1]!=2 or d.shape!=(len(q),32,2) or y.shape!=d.shape
            or w.shape!=d.shape[:2] or not all(np.isfinite(v).all() for v in (q,d,y,w))
            or (w<0).any() or (w.sum(1)<=0).any()):
        raise ValueError('Query predictions and source-excluded donor data only')
    w=w/w.sum(1,keepdims=True)
    return q+np.einsum('nk,nkc->nc',w,y-d)
