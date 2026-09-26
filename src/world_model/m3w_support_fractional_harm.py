"""Causal-support fractional-harm supervision, not occurrence calibration."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from src.world_model import m3w_selected_risk_learning as base
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch


def fractional_targets(y, envelope):
    y,d=np.asarray(y,float),np.asarray(envelope,float)
    if y.shape!=(len(d),4) or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Aligned causal envelope and four targets required')
    known=np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1),~known) or (y[known]<0).any():
        raise ValueError('Paired known/unknown nonnegative targets required')
    support=known&(d>0); q=np.zeros((len(y),2))
    if (y[known&(d==0)][:,[1,3]]!=0).any():
        raise ValueError('Zero envelope cannot carry observed harm')
    q[support]=y[support][:,[1,3]]/d[support,None]
    if not np.isfinite(q).all() or (q>1+1e-5).any():
        raise ValueError('Harm exceeds causal envelope')
    roundoff=int((q>1).sum()); q=np.minimum(q,1)
    return q,support,roundoff


def auxiliary(pred,envelope,q,sites):
    active=envelope>0; terms=[]
    for site in sorted(set(sites)):
        take=torch.as_tensor(sites==site)&active
        if take.any():
            probability=(pred[take][:,[1,3]]/envelope[take,None]).clamp(1e-7,1-1e-7)
            terms.append(F.binary_cross_entropy(probability,q[take].detach()))
    return (torch.stack(terms).mean() if terms else pred.sum()*0),int(active.sum()),len(terms)


def fit(x,y,sites,env,masks,pr,*,seed,settings,identity,directory,heartbeat,
        coefficient=1.,resume=False,stop_at=None):
    if not np.isfinite(coefficient) or coefficient<0: raise ValueError('Nonnegative fixed coefficient required')
    q,support,roundoff=fractional_targets(y,env); known=np.isfinite(y).all(1)
    np.testing.assert_array_equal(known,pr['known'])
    if masks.dtype!=bool or masks.shape!=(len(x),3): raise ValueError('Aligned causal masks required')
    w,scale=pr['weights'],pr['cost_scale']
    target=torch.from_numpy(np.where(known[:,None],y/scale,0).astype(np.float32))
    means=np.sum(w[:,None]*target.numpy(),axis=0)
    rms=np.sqrt(np.sum(w[:,None]*target.numpy().astype(float)**2,axis=0)).clip(1e-4)
    loss_scales=torch.tensor(rms,dtype=torch.float32)
    torch.manual_seed(seed); model=base.EventMomentHead(x.shape[1],settings['width'])
    mean_env=float(np.dot(w,env/scale))
    def logit(v):
        v=np.clip(v,1e-6,1-1e-6); return np.log(v/(1-v))
    ref=max(means[0],1e-6)
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([ref+np.log(-np.expm1(-ref)),
            logit(means[1]/max(mean_env,1e-6)),logit(means[2]/max(means[0],1e-6)),
            logit(means[3]/max(means[1],1e-6))],dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(x,pr)); d=torch.tensor(env/scale,dtype=torch.float32)
    if not np.array_equal((d.numpy()>0),env>0): raise FloatingPointError('Causal support lost in float32 conversion')
    qt=torch.tensor(q,dtype=torch.float32); mt=torch.as_tensor(masks,dtype=torch.bool)
    groups=[np.flatnonzero(known&(sites==s)) for s in sorted(set(sites))]
    rng=torch.Generator().manual_seed(seed+7919)
    fixed_rng=torch.Generator().set_state(rng.get_state()); fixed=draw_batch(groups,settings['batch_size'],fixed_rng)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step,seconds,trace,draws=0,0.,[],np.zeros(len(x),np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in dict(identity=identity,seed=seed,settings=settings,coefficient=coefficient).items():
            if saved[k]!=v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean','std','known','weights'): np.testing.assert_array_equal(saved['preprocess'][k],pr[k])
        assert saved['preprocess']['cost_scale']==scale
        np.testing.assert_array_equal(saved['loss_scales'],rms); np.testing.assert_array_equal(saved['fixed_ids'],fixed)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step,seconds,trace,draws=saved['step'],saved['seconds'],saved['trace'],saved['draws']
    limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<step or limit<=0: raise ValueError('Nondecreasing budget required')
    started=time.monotonic(); first=step
    def loss_for(ids):
        pred=model(z[ids],d[ids]); mse,parts=base.objective(pred,target[ids],loss_scales,mt[ids],sites[ids],'mean')
        aux,n,localities=auxiliary(pred,d[ids],qt[ids],sites[ids])
        return (mse if coefficient==0 else mse+coefficient*aux),dict(parts,
            fractional_BCE=float(aux.detach()),support_rows=n,support_localities=localities)
    def diagnostic():
        with torch.no_grad(): loss,parts=loss_for(fixed)
        return dict(step=step,loss=float(loss),**parts)
    def save():
        value=dict(identity=identity,seed=seed,settings=settings,coefficient=coefficient,preprocess=pr,
            loss_scales=rms,fixed_ids=fixed,step=step,seconds=seconds+time.monotonic()-started,
            model=model.state_dict(),optimizer=optimizer.state_dict(),sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(),draws=draws,trace=trace)
        tmp=path.with_suffix('.tmp'); torch.save(value,tmp); os.replace(tmp,path)
    if not trace: trace.append(diagnostic())
    model.train()
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng); optimizer.zero_grad(set_to_none=True)
            loss,parts=loss_for(ids)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fractional objective')
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); step+=1; np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(diagnostic(),batch_loss=float(loss.detach()),gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_checkpoint',step=step); raise
    model.eval()
    return model,dict(step=step,complete=step==settings['steps'],new_updates=step-first,
        seconds=seconds+time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),
        unknown_rows_sampled=int(draws[~known].sum()),support_draws=int(draws[support].sum()),
        total_draws=int(draws.sum()),unique_training_rows=int((draws>0).sum()),
        target_roundoff_clamps=roundoff,coefficient=coefficient,trace=trace,loss_scales=rms.tolist())
