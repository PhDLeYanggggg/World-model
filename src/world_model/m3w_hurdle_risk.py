"""Separate event-harm occurrence and severity without future inference inputs."""
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized, cost_loss


def factor_targets(y, envelope):
    y,d=np.asarray(y,float),np.asarray(envelope,float)
    if (y.ndim!=2 or y.shape[1]!=2 or d.shape!=(len(y),) or not np.isfinite(d).all()
            or (d<0).any() or np.isinf(y).any()
            or not np.array_equal(np.isnan(y[:,0]),np.isnan(y[:,1]))):
        raise ValueError('Aligned supported event moments and causal envelope required')
    known=np.isfinite(y[:,0]);positive=known & (y[:,1]>0)
    if (y[known]<0).any() or (y[known,1]>d[known]+1e-5).any() or (d[positive]<=0).any():
        raise ValueError('Nonnegative harm labels must respect the causal bound')
    fraction=np.zeros(len(y));fraction[positive]=(y[positive,1]/d[positive]).clip(0,1)
    return dict(known=known,positive=positive,fraction=fraction)


class HurdleRiskHead(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(features,width),nn.GELU(),nn.Linear(width,3))

    def components(self,x,envelope):
        if envelope.shape!=(len(x),) or not torch.isfinite(envelope).all() or (envelope<0).any():
            raise ValueError('Finite nonnegative causal envelope required')
        raw=self.network(x)
        probability,severity=torch.sigmoid(raw[:,1]),torch.sigmoid(raw[:,2])
        moments=torch.stack((F.softplus(raw[:,0]),envelope*probability*severity),1)
        return dict(moments=moments,probability=probability,severity=severity,logit=raw[:,1])

    def forward(self,x,envelope):
        return self.components(x,envelope)['moments']


def objective(parts,target,envelope,arm):
    if arm not in ('product_mse','hurdle'):
        raise ValueError('Registered objective required')
    moment=cost_loss(parts['moments'],target,'mse')
    positive=target[:,1]>0
    bce=F.binary_cross_entropy_with_logits(parts['logit'],positive.float())
    # An empty positive subset has no severity supervision, not a zero target.
    conditional=((parts['severity'][positive]-target[positive,1]/envelope[positive]).square().mean()
        if positive.any() else parts['severity'].sum()*0)
    loss=moment+(bce+conditional if arm=='hurdle' else 0)
    return loss,dict(moment_mse=float(moment.detach()),bce=float(bce.detach()),
        conditional_mse=float(conditional.detach()),positive_rows=int(positive.sum()))


def initialize(pr,prior,width,seed):
    torch.manual_seed(seed)
    model=HurdleRiskHead(len(pr['mean']),width)
    ref=max(pr['constant'][0]/pr['cost_scale'],1e-6)
    p,q=np.clip([prior['probability'],prior['severity']],1e-6,1-1e-6)
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor(
            [ref+np.log(-np.expm1(-ref)),np.log(p/(1-p)),np.log(q/(1-q))],dtype=torch.float32))
    return model


def fit(x,y,sites,envelope,pr,*,seed,arm,settings,identity,directory,heartbeat,resume=False,stop_at=None):
    d=np.asarray(envelope,float);f=factor_targets(y,d);known=f['known']
    np.testing.assert_array_equal(known,pr['known'])
    w=np.asarray(pr['weights']);p=float(np.dot(w,f['positive']))
    prior=dict(probability=p,severity=float(np.dot(w,f['fraction'])/p) if p>0 else 0.)
    model=initialize(pr,prior,settings['width'],seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(x,pr))
    dt=torch.from_numpy((d/pr['cost_scale']).astype(np.float32))
    target=torch.from_numpy(np.where(known[:,None],y/pr['cost_scale'],0).astype(np.float32))
    groups=[np.flatnonzero(known & (sites==s)) for s in sorted(set(sites))]
    rng=torch.Generator().manual_seed(seed+7919)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'checkpoint.pt'
    step,seconds,trace,draws=0,0.,[],np.zeros(len(x),np.int64)
    if path.exists():
        if not resume:raise ValueError('Existing checkpoint requires explicit resume')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['settings']!=settings or state['arm']!=arm or state['seed']!=seed or state['prior']!=prior:
            raise ValueError('Resume identity mismatch')
        for k in ('mean','std','constant','weights','known'):
            np.testing.assert_array_equal(state['preprocess'][k],pr[k])
        if state['preprocess']['cost_scale']!=pr['cost_scale']:raise ValueError('Changed training scale')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']);torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,draws=state['step'],state['seconds'],state['trace'],state['draws']
    first,started=step,time.monotonic()
    limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<step or limit<=0:raise ValueError('Nondecreasing fixed training budget required')
    def save():
        state=dict(identity=identity,settings=settings,arm=arm,seed=seed,preprocess=pr,prior=prior,
            model=model.state_dict(),optimizer=optimizer.state_dict(),sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(),draws=draws,step=step,seconds=seconds+time.monotonic()-started,trace=trace)
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
    model.train()
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng)
            optimizer.zero_grad(set_to_none=True)
            loss,components=objective(model.components(z[ids],dt[ids]),target[ids],dt[ids],arm)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hurdle objective')
            loss.backward()
            grad=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();step+=1;np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),**components)
                trace.append(row);heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit:save()
    except BaseException:
        heartbeat(state='interrupted_resume_atomic_checkpoint',step=step)
        raise
    model.eval()
    return model,dict(step=step,new_updates=step-first,complete=step==settings['steps'],
        seconds=seconds+time.monotonic()-started,total_draws=int(draws.sum()),
        unique_training_rows=int((draws>0).sum()),unknown_rows_sampled=int(draws[~known].sum()),
        parameters=sum(p.numel() for p in model.parameters()),prior=prior,trace=trace)


def predict(model,x,envelope,pr,*,components=False):
    out=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z=torch.from_numpy(standardized(x[start:start+4096],pr))
            d=torch.as_tensor(np.asarray(envelope[start:start+4096])/pr['cost_scale'],dtype=torch.float32)
            v=model.components(z,d);moment=v['moments'].numpy()*pr['cost_scale']
            out.append(np.column_stack((moment,v['probability'].numpy(),v['severity'].numpy())) if components else moment)
    return np.concatenate(out)
