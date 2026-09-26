"""Past-input membership probabilities; realized baseline error is supervision only."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch


def labels(cv, cut):
    cv=np.asarray(cv,float)
    if cv.ndim!=1 or np.isinf(cv).any() or (cv[np.isfinite(cv)]<0).any() or not np.isfinite(cut) or cut<=0:
        raise ValueError('Nonnegative baseline costs and positive fitting-only cut required')
    return np.where(np.isfinite(cv),((cv>0)&(cv<=cut)).astype(float),np.nan)


class MembershipHead(nn.Module):
    def __init__(self,dimension,arm,width=64):
        super().__init__()
        if arm=='linear': self.network=nn.Sequential(nn.Linear(dimension,1))
        elif arm=='mlp': self.network=nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Linear(width,1))
        else: raise ValueError('Registered linear or MLP arm required')

    def forward(self,x): return self.network(x).squeeze(-1)


def fit(x,y,sites,pr,*,arm,seed,settings,identity,directory,heartbeat,resume=False,stop_at=None):
    known=np.isfinite(y)
    if y.shape!=(len(x),) or not np.array_equal(known,pr['known']) or not np.isin(y[known],[0,1]).all():
        raise ValueError('Aligned binary/unknown membership labels required')
    if not np.isfinite(x).all(): raise ValueError('Finite causal inputs required')
    prevalence=float(pr['weights']@np.nan_to_num(y,nan=0.))
    torch.manual_seed(seed); model=MembershipHead(x.shape[1],arm,settings['width'])
    with torch.no_grad():
        model.network[-1].weight.zero_()
        p=np.clip(prevalence,1e-6,1-1e-6); model.network[-1].bias.fill_(float(np.log(p/(1-p))))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(x,pr)); target=torch.tensor(np.nan_to_num(y,nan=0.),dtype=torch.float32)
    groups=[np.flatnonzero(known&(sites==s)) for s in sorted(set(sites))]
    if any(not len(g) for g in groups): raise ValueError('Each fitting locality needs known labels')
    rng=torch.Generator().manual_seed(seed+7919); fixed_rng=torch.Generator().set_state(rng.get_state())
    fixed=draw_batch(groups,settings['batch_size'],fixed_rng)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step,seconds,trace,draws=0,0.,[],np.zeros(len(x),np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        s=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in dict(identity=identity,seed=seed,arm=arm,settings=settings,prevalence=prevalence).items():
            if s[k]!=v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean','std','known','weights'): np.testing.assert_array_equal(pr[k],s['preprocess'][k])
        np.testing.assert_array_equal(s['fixed_ids'],fixed)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step,seconds,trace,draws=s['step'],s['seconds'],s['trace'],s['draws']
    limit=settings['steps'] if stop_at is None else min(settings['steps'],stop_at)
    if limit<=0 or limit<step: raise ValueError('Nondecreasing budget required')
    first=step; started=time.monotonic()
    def diagnostic():
        with torch.no_grad():
            logits=model(z[fixed]); p=logits.sigmoid(); t=target[fixed]
            return dict(step=step,BCE=float(nn.functional.binary_cross_entropy_with_logits(logits,t)),Brier=float(((p-t)**2).mean()))
    def save():
        s=dict(identity=identity,seed=seed,arm=arm,settings=settings,prevalence=prevalence,preprocess=pr,
            step=step,seconds=seconds+time.monotonic()-started,trace=trace,draws=draws,fixed_ids=fixed,
            model=model.state_dict(),optimizer=optimizer.state_dict(),sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state())
        tmp=path.with_suffix('.tmp'); torch.save(s,tmp); os.replace(tmp,path)
    if not trace: trace.append(diagnostic())
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng); optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(z[ids]),target[ids])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite membership loss')
            loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); step+=1; np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(diagnostic(),batch_loss=float(loss.detach()),gradient_norm=float(grad)); trace.append(row); heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint',step=step); raise
    model.eval()
    return model,dict(step=step,new_updates=step-first,complete=step==settings['steps'],
        seconds=seconds+time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),
        prevalence=prevalence,unknown_rows_sampled=int(draws[~known].sum()),trace=trace)


def predict(model,x,pr):
    out=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z=torch.from_numpy(standardized(x[start:start+4096],pr)); out.append(model(z).sigmoid().numpy())
    p=np.concatenate(out)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise FloatingPointError('Invalid probability')
    return p


def restore(directory):
    s=torch.load(Path(directory)/'checkpoint.pt',map_location='cpu',weights_only=False)
    model=MembershipHead(len(s['preprocess']['mean']),s['arm'],s['settings']['width'])
    model.load_state_dict(s['model']); model.eval(); return model,s
