"""Objective-only fixed-floor contrast; output components are not identified moments."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_fixed_floor_tail import initialize, predict
from src.world_model.m3w_native_forecast import draw_batch
from scripts.replay_m3w_dimensionless_training import exact


def signed(parts):
    return parts[:,[1,3]]-.02*parts[:,[0,2]]


def loss(prediction,target):
    if (prediction.shape!=target.shape or target.ndim!=2 or target.shape[1]!=4
            or not torch.isfinite(target).all() or (target<0).any()):
        raise ValueError('Four finite nonnegative fitting-only targets required')
    return (signed(prediction)-signed(target.detach())).square().mean()


def assert_matched(a,b):
    for k in ('settings','preprocess','seed','step','draws','sampler_rng','torch_rng','initial_model','mean_envelope'):
        exact(a[k],b[k])
    assert a['objective']=='fixed_floor_signed_excess' and b['arm']=='mse'


def fit(x,y,sites,envelope,pr,*,seed,settings,identity,directory,heartbeat,resume=False,stop_at=None):
    x,y,sites,env=map(np.asarray,(x,y,sites,envelope)); known=pr['known']
    if (not np.array_equal(np.isfinite(y).all(1),known) or set(sites)!=set(pr['training_sites'])
            or not np.isfinite(env).all() or (env<0).any() or (y[known][:,[1,3]]>env[known,None]+1e-5).any()):
        raise ValueError('Matched fitting-only support and causal envelope required')
    mean_envelope=float(np.dot(pr['weights'],env))
    model=initialize(pr,settings['width'],seed,mean_envelope)
    initial_model={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(np.clip((x.astype(float)-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32))
    target=torch.from_numpy(np.where(known[:,None],(y/pr['cost_scale']).astype(np.float32),0).astype(np.float32))
    dt=torch.from_numpy((env/pr['cost_scale']).astype(np.float32))
    groups=[np.flatnonzero(known&(sites==s)) for s in sorted(set(sites))]
    rng=torch.Generator().manual_seed(seed+7919)
    monitor=draw_batch(groups,1024,torch.Generator().manual_seed(seed+9137))
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step=0; seconds=0.; trace=[]; draws=np.zeros(len(x),np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing head needs --resume')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if (state['identity']!=identity or state['settings']!=settings or state['seed']!=seed
                or state['objective']!='fixed_floor_signed_excess'):
            raise ValueError('Resume changes experiment or fixed budget')
        for a,b in ((state['preprocess'],pr),(state['initial_model'],initial_model),(state['mean_envelope'],mean_envelope)): exact(a,b)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,draws=state['step'],state['seconds'],state['trace'],state['draws']
    first=step; start=time.monotonic(); limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<step or limit<=0: raise ValueError('Nondecreasing step limit required')
    def save():
        state=dict(identity=identity,settings=settings,objective='fixed_floor_signed_excess',seed=seed,preprocess=pr,
            model=model.state_dict(),initial_model=initial_model,optimizer=opt.state_dict(),sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(),draws=draws,step=step,seconds=seconds+time.monotonic()-start,
            trace=trace,mean_envelope=mean_envelope)
        temp=path.with_suffix('.tmp'); torch.save(state,temp); os.replace(temp,path)
    model.train()
    try:
        while step<limit:
            ix=draw_batch(groups,settings['batch_size'],rng)
            opt.zero_grad(set_to_none=True); p=model(z[ix],dt[ix]); objective=loss(p,target[ix])
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite signed-excess loss')
            objective.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            opt.step(); step+=1; np.add.at(draws,ix,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==settings['steps']:
                with torch.no_grad():
                    pred=model(z[monitor],dt[monitor]); fixed=float(loss(pred,target[monitor]))
                row=dict(step=step,loss=float(objective.detach()),fixed_training_excess_MSE=fixed,gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint',step=step)
        raise
    return model,dict(step=step,new_updates=step-first,complete=step==settings['steps'],
        seconds=seconds+time.monotonic()-start,total_draws=int(draws.sum()),unknown_rows_sampled=int(draws[~known].sum()),
        trace=trace,parameters=sum(p.numel() for p in model.parameters()),objective='fixed_floor_signed_excess',
        component_moments_identified=False)
