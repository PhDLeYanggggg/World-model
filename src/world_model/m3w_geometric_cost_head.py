"""Candidate-conditioned cost moments bounded by a past-only rollout envelope."""
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized, cost_loss


def rollout_envelope(baseline, candidate):
    b,p = np.asarray(baseline,float),np.asarray(candidate,float)
    if b.ndim!=3 or b.shape[1:]!=(12,2) or p.shape!=b.shape or not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('Aligned finite twelve-step causal forecasts required')
    # Max, not mean: future labels may cover any subset of requested steps.
    return np.linalg.norm(p-b,axis=-1).max(1)


class EnvelopeCostHead(nn.Module):
    def __init__(self, features, width, task):
        super().__init__()
        if task not in ('utility','all','easy'):
            raise ValueError('Explicit utility or event-moment task required')
        self.task=task
        self.network=nn.Sequential(nn.Linear(features,width),nn.GELU(),nn.Linear(width,2))

    def forward(self, x, envelope):
        if envelope.shape!=(len(x),) or not torch.isfinite(envelope).all() or (envelope<0).any():
            raise ValueError('Finite nonnegative causal envelope required')
        raw=self.network(x)
        if self.task=='utility':
            fractions=torch.softmax(torch.cat((raw,torch.zeros_like(raw[:,:1])),1),1)[:,:2]
            return envelope[:,None]*fractions
        return torch.stack((F.softplus(raw[:,0]),envelope*torch.sigmoid(raw[:,1])),1)


def initialize_head(width, pr, seed, task, mean_envelope):
    if not np.isfinite(mean_envelope) or mean_envelope<0 or pr['cost_scale']<=0:
        raise ValueError('Training-only finite scale required')
    constant=np.asarray(pr['constant'],float)
    if (constant.shape!=(2,) or not np.isfinite(constant).all() or (constant<0).any()
            or (constant.sum() if task=='utility' else constant[1])>mean_envelope+1e-5):
        raise ValueError('Training means violate the causal envelope')
    torch.manual_seed(seed)
    model=EnvelopeCostHead(len(pr['mean']),width,task)
    with torch.no_grad():
        layer=model.network[-1];layer.weight.zero_()
        if task=='utility':
            mass=np.maximum(np.r_[constant,max(0,mean_envelope-constant.sum())],1e-6)
            bias=np.log(mass[:2]/mass[2])
        else:
            initial=max(constant[0]/pr['cost_scale'],1e-6)
            fraction=np.clip(constant[1]/max(mean_envelope,1e-6),1e-6,1-1e-6)
            bias=np.array([initial+np.log(-np.expm1(-initial)),np.log(fraction/(1-fraction))])
        layer.bias.copy_(torch.as_tensor(bias,dtype=torch.float32))
    return model


def fit(x,y,sites,envelope,pr,*,seed,task,settings,identity,directory,heartbeat,resume=False,stop_at=None):
    d=np.asarray(envelope,float)
    if d.shape!=(len(x),) or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Aligned causal envelope required')
    known=pr['known']
    bounded=y[known].sum(1) if task=='utility' else y[known,1]
    if np.any(bounded>d[known]+1e-5):
        raise ValueError('Supervision exceeds future-mask-independent envelope')
    mean_envelope=float(np.dot(pr['weights'],d))
    model=initialize_head(settings['width'],pr,seed,task,mean_envelope)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(x,pr))
    dt=torch.from_numpy((d/pr['cost_scale']).astype(np.float32))
    target=torch.from_numpy(np.where(known[:,None],y/pr['cost_scale'],0).astype(np.float32))
    groups=[np.flatnonzero(known & (sites==s)) for s in sorted(set(sites))]
    rng=torch.Generator().manual_seed(seed+7919)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'checkpoint.pt'
    step,seconds,trace,draws=0,0.,[],np.zeros(len(x),np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['settings']!=settings or state['task']!=task or state['seed']!=seed:
            raise ValueError('Resume identity mismatch')
        for key in ('mean','std','constant','weights','known'):
            np.testing.assert_array_equal(pr[key],state['preprocess'][key])
        if mean_envelope!=state['mean_envelope'] or pr['cost_scale']!=state['preprocess']['cost_scale']:
            raise ValueError('Training normalizers changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']);torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,draws=state['step'],state['seconds'],state['trace'],state['draws']
    first,started=step,time.monotonic()
    limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<step or limit<=0:
        raise ValueError('Nondecreasing fixed update budget required')
    def save():
        state=dict(identity=identity,settings=settings,task=task,seed=seed,preprocess=pr,
            mean_envelope=mean_envelope,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),draws=draws,step=step,
            seconds=seconds+time.monotonic()-started,trace=trace)
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
    model.train()
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng)
            optimizer.zero_grad(set_to_none=True)
            p=model(z[ids],dt[ids]);loss=cost_loss(p,target[ids],'mse')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native cost loss')
            loss.backward()
            grad=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();step+=1;np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad))
                trace.append(row);heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_atomic_checkpoint',step=step)
        raise
    model.eval()
    return model,dict(step=step,new_updates=step-first,complete=step==settings['steps'],
        seconds=seconds+time.monotonic()-started,total_draws=int(draws.sum()),
        unique_training_rows=int((draws>0).sum()),unknown_rows_sampled=int(draws[~known].sum()),
        parameters=sum(p.numel() for p in model.parameters()),mean_envelope=mean_envelope,trace=trace)


def predict(model,x,envelope,pr):
    out=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z=torch.from_numpy(standardized(x[start:start+4096],pr))
            d=torch.as_tensor(np.asarray(envelope[start:start+4096])/pr['cost_scale'],dtype=torch.float32)
            out.append(model(z,d).numpy()*pr['cost_scale'])
    return np.concatenate(out)
