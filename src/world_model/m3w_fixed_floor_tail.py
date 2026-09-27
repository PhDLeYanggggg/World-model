"""Bounded conditional harm heads with a fitting-only tail-weighted contrast."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model.m3w_native_forecast import draw_batch


def tail_parameters(y,weights):
    y,w=np.asarray(y,float),np.asarray(weights,float)
    known=np.isfinite(y).all(1)
    if (y.ndim!=2 or y.shape[1]!=4 or w.shape!=(len(y),) or not np.isfinite(w).all()
            or (w<0).any() or np.any(w[~known]!=0) or not np.isclose(w.sum(),1)
            or (y[known]<0).any()):
        raise ValueError('Aligned nonnegative fitting targets and source weights required')
    thresholds=[]; normalizers=[]
    for col in (1,3):
        pos=known&(y[:,col]>0)&(w>0)
        if not pos.any(): thresholds.append(None); normalizers.append(1.); continue
        v=y[pos,col]; a=w[pos]; order=np.argsort(v,kind='stable')
        idx=min(np.searchsorted(np.cumsum(a[order])/a.sum(),.9),len(order)-1)
        threshold=float(v[order[idx]])
        raw=np.where(known&(y[:,col]>=threshold),4.,1.)
        thresholds.append(threshold); normalizers.append(float(np.dot(w,raw)))
    return dict(thresholds=thresholds,normalizers=normalizers,quantile=.9,multiplier=4.)


def tail_weights(target,parameters):
    target=target.detach(); weights=torch.ones_like(target)
    for i,col in enumerate((1,3)):
        cut=parameters['thresholds'][i]
        if cut is not None:
            weights[:,col]=torch.where(target[:,col]>=cut,4.,1.)/parameters['normalizers'][i]
    return weights


def loss(prediction,target,arm,parameters):
    if (arm not in ('mse','tail4') or prediction.shape!=target.shape or target.ndim!=2
            or target.shape[1]!=4 or not torch.isfinite(target).all() or (target<0).any()):
        raise ValueError('Four finite nonnegative fitting risk moments required')
    error=(prediction-target.detach()).square()
    return (error*(tail_weights(target,parameters) if arm=='tail4' else 1.)).mean()


class RiskHead(nn.Module):
    def __init__(self,features,width):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(features,width),nn.GELU(),nn.Linear(width,4))

    def forward(self,x,envelope):
        raw=self.network(x)
        return torch.stack((F.softplus(raw[:,0]),envelope*torch.sigmoid(raw[:,1]),
                            F.softplus(raw[:,2]),envelope*torch.sigmoid(raw[:,3])),1)


def initialize(pr,width,seed,mean_envelope):
    torch.manual_seed(seed); model=RiskHead(len(pr['mean']),width)
    b=[]
    for col in range(4):
        if col%2==0:
            v=max(pr['constant'][col]/pr['cost_scale'],1e-6)
            b.append(v+np.log(-np.expm1(-v)))
        else:
            f=np.clip(pr['constant'][col]/max(mean_envelope,1e-6),1e-6,1-1e-6)
            b.append(np.log(f/(1-f)))
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor(b,dtype=torch.float32))
    return model


def predict(model,x,envelope,pr):
    x=np.asarray(x,float); env=np.asarray(envelope,float)
    if x.ndim!=2 or env.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(env).all() or (env<0).any():
        raise ValueError('Finite causal features and envelope only')
    out=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z=np.clip((x[start:start+4096]-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32)
            p=model(torch.from_numpy(z),torch.from_numpy((env[start:start+4096]/pr['cost_scale']).astype(np.float32)))
            out.append((p.numpy()*pr['cost_scale']).astype(np.float32))
    return np.concatenate(out)


def fit(x,y,sites,envelope,pr,*,seed,arm,settings,identity,directory,heartbeat,resume=False,stop_at=None):
    x,y,sites,env=np.asarray(x),np.asarray(y),np.asarray(sites),np.asarray(envelope)
    known=pr['known']
    if (not np.array_equal(np.isfinite(y).all(1),known) or set(sites)!=set(pr['training_sites'])
            or (y[known][:,[1,3]]>env[known,None]+1e-5).any()):
        raise ValueError('Training-only matched support and causal envelope required')
    # Quantiles stay in normalized target units, with weights from fitting sources only.
    target_np=(y/pr['cost_scale']).astype(np.float32)
    tail=tail_parameters(target_np,pr['weights'])
    mean_envelope=float(np.dot(pr['weights'],env))
    model=initialize(pr,settings['width'],seed,mean_envelope)
    initial_model={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(np.clip((x.astype(float)-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32))
    target=torch.from_numpy(np.where(known[:,None],target_np,0).astype(np.float32))
    dt=torch.from_numpy((env/pr['cost_scale']).astype(np.float32))
    groups=[np.flatnonzero(known&(sites==s)) for s in sorted(set(sites))]
    rng=torch.Generator().manual_seed(seed+7919)
    monitor=draw_batch(groups,1024,torch.Generator().manual_seed(seed+9137))
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step=0; seconds=0.; trace=[]; draws=np.zeros(len(x),np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing head needs --resume')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['settings']!=settings or state['arm']!=arm or state['seed']!=seed:
            raise ValueError('Resume changes identity/objective/budget')
        from scripts.replay_m3w_dimensionless_training import exact
        for old,new in ((state['preprocess'],pr),(state['tail'],tail),(state['initial_model'],initial_model)):
            exact(old,new)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,draws=state['step'],state['seconds'],state['trace'],state['draws']
    first=step; start=time.monotonic(); limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<step or limit<=0: raise ValueError('Nondecreasing step limit required')
    def save():
        state=dict(identity=identity,settings=settings,arm=arm,seed=seed,preprocess=pr,tail=tail,
            model=model.state_dict(),initial_model=initial_model,optimizer=opt.state_dict(),
            sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),draws=draws,step=step,
            seconds=seconds+time.monotonic()-start,trace=trace,mean_envelope=mean_envelope)
        temp=path.with_suffix('.tmp'); torch.save(state,temp); os.replace(temp,path)
    model.train()
    try:
        while step<limit:
            ix=draw_batch(groups,settings['batch_size'],rng)
            opt.zero_grad(set_to_none=True); p=model(z[ix],dt[ix]); objective=loss(p,target[ix],arm,tail)
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite training loss')
            objective.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            opt.step(); step+=1; np.add.at(draws,ix,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==settings['steps']:
                with torch.no_grad():
                    pred=model(z[monitor],dt[monitor])
                    fixed_mse=float(loss(pred,target[monitor],'mse',tail))
                    fixed_tail=float(loss(pred,target[monitor],'tail4',tail))
                row=dict(step=step,loss=float(objective.detach()),fixed_training_MSE=fixed_mse,
                         fixed_training_tail_loss=fixed_tail,gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint',step=step)
        raise
    return model,dict(step=step,new_updates=step-first,complete=step==settings['steps'],
        seconds=seconds+time.monotonic()-start,total_draws=int(draws.sum()),unknown_rows_sampled=int(draws[~known].sum()),
        trace=trace,parameters=sum(v.numel() for v in model.parameters()),objective=arm,
        expected_moment_calibration_claim=False)


def match_counts(anchor,eligible,score,recordings,frames,ids):
    anchor,eligible,score,recordings,frames,ids=map(np.asarray,(anchor,eligible,score,recordings,frames,ids))
    if (anchor.dtype!=bool or eligible.dtype!=bool or any(v.shape!=anchor.shape for v in (eligible,score,recordings,frames,ids))
            or np.any(anchor&~eligible) or not np.isfinite(score).all()):
        raise ValueError('Aligned causal eligibility and subset anchor required')
    result=np.zeros(len(ids),bool)
    keys=np.rec.fromarrays((recordings,frames),names=('recording','frame'))
    order=np.argsort(keys,kind='stable'); sorted_keys=keys[order]
    breaks=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1,len(order)]
    for first,last in zip(breaks[:-1],breaks[1:]):
        query=order[first:last]; k=int(anchor[query].sum())
        candidates=query[eligible[query]]
        ranked=candidates[np.lexsort((ids[candidates],score[candidates]))]
        result[ranked[:k]]=True
        assert result[query].sum()==k
    return result
