"""Action-invariant reference learning with a frozen expert's causal context."""
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_role_separated_cost as parent

ARMS = ('history_refit', 'expert_context')
SETTINGS = dict(steps=4096, batch_size=512, learning_rate=.001, weight_decay=.0001,
                gradient_clip=5., checkpoint_steps=128, log_every=256, context_width=70)
base = parent.storage.storage.base


def context_table(x, ids):
    """Full-amplitude descriptors are shared per agent; no outcome API exists."""
    x, ids = np.asarray(x), np.asarray(ids)
    n = len(ids)
    if (x.shape != (n,120) or not np.isfinite(x).all() or not n
            or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any()
            or len(np.unique(ids))!=n or not np.isin(ids%7,np.arange(1,7)).all()):
        raise ValueError('Unique original legal row/action IDs and finite causal120 required')
    order=np.argsort(ids);agents=ids//7;wanted=agents*7+6
    at=np.searchsorted(ids[order],wanted)
    if (at==n).any() or not np.array_equal(ids[order[at]],wanted):
        raise ValueError('Every candidate requires its original full-amplitude expert anchor')
    anchor=order[at]
    np.testing.assert_array_equal(x[:,:50],x[anchor,:50])
    unique,first=np.unique(agents,return_index=True)
    return dict(ids=unique,context=x[anchor[first],50:].astype(np.float32))


def lookup(table, ids):
    ids=np.asarray(ids);agents=ids//7;at=np.searchsorted(table['ids'],agents)
    if (at==len(table['ids'])).any() or not np.array_equal(table['ids'][at],agents):
        raise ValueError('Exact agent alignment; never nearest-row or future-label joins')
    return table['context'][at]


def prepare(data, held, origin):
    p=parent.prepare(data,held)
    base.exact(p['binding'],origin['preprocess'])
    if origin['arm']!='role_separated' or not origin['complete'] or origin['step']!=4096:
        raise ValueError('Complete immutable role-separated parent required')
    c=np.asarray(data['context']);fit=data['sites']!=held;known=fit&data['observed']
    if c.shape!=(len(data['ids']),70) or not np.isfinite(c).all():
        raise ValueError('Finite action-invariant frozen-expert context only')
    w=data['weights'][known];w=w/w.sum()
    mean=np.einsum('n,nd->d',w,c[known]);std=np.sqrt(np.einsum('n,nd->d',w,(c[known]-mean)**2)).clip(1e-6)
    context=np.clip((c[fit]-mean)/std,-8,8).astype(np.float32)
    binding=dict(parent_preprocess=p['binding'],context_mean=mean,context_std=std,
                 context_fit_hash=parent.storage.digest(c[fit]),train_ids=data['ids'][fit],
                 held_source=held,context_normalization_known_fitting_only=True)
    return dict(p,context=torch.from_numpy(context),binding=binding)


class Reference(nn.Module):
    def __init__(self, origin):
        super().__init__()
        self.history=nn.Linear(50,32)
        self.context=nn.Linear(70,32,bias=False)
        self.output=nn.Linear(32,2)
        state=origin['model']
        with torch.no_grad():
            self.history.weight.copy_(state['reference.0.weight'])
            self.history.bias.copy_(state['reference.0.bias'])
            self.output.weight.copy_(state['reference.2.weight'])
            self.output.bias.copy_(state['reference.2.bias'])
            self.context.weight.zero_()

    def forward(self, history, context, arm):
        if history.shape!=(len(context),50) or context.shape[1:]!=(70,) or arm not in ARMS:
            raise ValueError('Past50, shared expert70 and fixed feature arm required')
        z=context if arm=='expert_context' else torch.zeros_like(context)
        raw=self.output(F.silu(self.history(history)+self.context(z)))
        reference=F.softplus(raw[:,0])
        return torch.stack((reference,reference*torch.sigmoid(raw[:,1])),1)


def fit(data,held,origin,identity,path,*,arm,seed,resume=False,stop_at=None,
        guard=lambda:None,heartbeat=lambda **_:None):
    if arm not in ARMS or seed not in (17,29,43):raise ValueError('Fixed matched arms/seeds required')
    p=prepare(data,held,origin);torch.manual_seed(seed);model=Reference(origin)
    initial=copy.deepcopy(model.state_dict());cfg=SETTINGS
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=torch.Generator().manual_seed(seed);n=len(p['x'])
    binding=dict(identity=identity,settings=cfg,arm=arm,seed=seed,preprocess=p['binding'],
                 initial_model=initial,parameters=sum(v.numel() for v in model.parameters()),fit_rows=n)
    step,elapsed,trace,visits=0,0.,[],np.zeros(n,np.int32);path=Path(path)
    if path.exists():
        if not resume:raise ValueError('Explicit same-definition resume required')
        old=base.read_checkpoint(path)
        for k,v in binding.items():base.exact(old[k],v)
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
        rng.set_state(old['generator']);torch.set_rng_state(old['torch_rng'])
        step,elapsed,trace,visits=(old[k] for k in ('step','seconds','trace','visits'))
    limit=cfg['steps'] if stop_at is None else stop_at
    if not step<=limit<=cfg['steps']:raise ValueError('Nondecreasing registered budget only')
    start=time.monotonic()
    def save():
        path.parent.mkdir(parents=True,exist_ok=True)
        base.save_checkpoint(path,dict(**binding,model=model.state_dict(),optimizer=opt.state_dict(),
            generator=rng.get_state(),torch_rng=torch.get_rng_state(),step=step,seconds=elapsed+time.monotonic()-start,
            trace=trace,visits=visits,complete=step==cfg['steps']))
    try:
        while step<limit:
            guard();ix=torch.multinomial(p['probability'],cfg['batch_size'],replacement=True,generator=rng)
            opt.zero_grad(set_to_none=True)
            pred=model(p['x'][ix,:50],p['context'][ix],arm)
            error=(((pred-p['y'][ix][:,[2,3]])/p['rms'][[2,3]])**2).mean(1)
            known=p['known'][ix];loss=(error*known).sum()/known.sum().clamp_min(1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite observed-reference loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            opt.step();step+=1;np.add.at(visits,ix.numpy(),1)
            if step==1 or step%cfg['log_every']==0:
                row=dict(step=step,reference_MSE=float(loss.detach()),gradient_norm=float(norm),
                         observed_draws=int(known.sum()),distinct_fitting_rows_drawn=int((visits>0).sum()))
                trace.append(row);heartbeat(state='reference_training',arm=arm,**row)
            if step%cfg['checkpoint_steps']==0 or step==limit:save()
    except BaseException:
        save();raise
    return base.read_checkpoint(path)


def predict(state, origin, candidate_x, shared_context):
    x,c=np.asarray(candidate_x),np.asarray(shared_context)
    if x.shape!=(len(c),120) or c.shape!=(len(x),70) or not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError('Causal candidate120 and shared expert70 only')
    p=state['preprocess'];old=p['parent_preprocess'];model=Reference(origin)
    model.load_state_dict(state['model']);model.eval()
    x=np.clip((x[:,:50]-old['mean'][:50])/old['std'][:50],-8,8).astype(np.float32)
    c=np.clip((c-p['context_mean'])/p['context_std'],-8,8).astype(np.float32)
    result=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            result.append(model(torch.from_numpy(x[start:start+512]),torch.from_numpy(c[start:start+512]),state['arm']).numpy())
    return np.concatenate(result).astype(float)


def replace_reference(frozen, references, allowed):
    c=np.asarray(frozen['conditional']);p=np.asarray(frozen['moments']);q=np.asarray(frozen['probability'])
    r=np.asarray(references);ok=np.asarray(allowed)
    if (c.shape!=p.shape or p.ndim!=3 or p.shape[1:]!=(7,5) or q.shape!=p.shape[:2]
            or r.shape!=(*q.shape,2) or ok.shape!=q.shape or ok.dtype!=bool
            or not all(np.isfinite(v).all() for v in (c,p,q,r)) or (r<0).any() or (r[...,1]>r[...,0]).any()):
        raise ValueError('Aligned coherent reference replacements and frozen moment bank required')
    out=c.copy();out[...,2:4][ok]=r[ok];out[...,0][ok]=np.minimum(c[...,0][ok],r[...,0][ok])
    composed=p.copy()
    for j in (0,2,3):composed[...,j][ok]=q[ok]*out[...,j][ok]
    # Preserve frozen harm/support exactly; only the shared reference and required B cap change.
    for j in (1,4):np.testing.assert_array_equal(composed[...,j],p[...,j])
    return dict(conditional=out,moments=composed,probability=q.copy())


def exact(a,b):
    for k in ('identity','settings','arm','seed','preprocess','initial_model','model','optimizer',
              'generator','torch_rng','step','trace','visits','complete','fit_rows'):
        base.exact(a[k],b[k])
