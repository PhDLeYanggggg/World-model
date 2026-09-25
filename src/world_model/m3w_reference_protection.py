"""Matched continuation with an optional immutable reference-moment branch."""
from copy import deepcopy
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.world_model.m3w_selected_risk_learning import EventMomentHead
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized

ARMS = ('continued','protected')


class ContinuationHead(nn.Module):
    def __init__(self, initial, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError('Unknown continuation arm')
        self.arm = arm
        self.learned = deepcopy(initial)
        self.reference = deepcopy(initial) if arm == 'protected' else None
        if self.reference is not None:
            self.reference.requires_grad_(False)

    def forward(self, x, envelope):
        value = self.learned(x,envelope)
        if self.reference is None: return value
        with torch.no_grad(): reference = self.reference(x,envelope)
        return torch.stack((reference[:,0],value[:,1],reference[:,2],value[:,3]),dim=1)


def loss_parts(prediction, target, scales):
    if (prediction.shape != target.shape or prediction.shape[1] != 4 or
        scales.shape != (4,) or not torch.isfinite(prediction).all() or
        not torch.isfinite(target).all() or not torch.isfinite(scales).all() or (scales<=0).any()):
        raise ValueError('Finite matched moments and B-only loss scales required')
    error = (prediction-target.detach())/scales
    components = error.square().mean(0)
    loss = error.square().mean()
    return loss,dict(moment_mse=float(loss.detach()),component_mse=components.detach().tolist())


def fit(x,y,sites,env,pr,initial,control,*,arm,seed,settings,identity,directory,
        heartbeat,resume=False,stop_at=None):
    known=pr['known']; np.testing.assert_array_equal(known,np.isfinite(y).all(1))
    if (y[known]<0).any() or (y[known,1]>env[known]+1e-5).any() or (y[known,2:]>y[known,:2]+1e-5).any():
        raise ValueError('Invalid nested targets')
    for key in ('mean','std','weights','known'):
        np.testing.assert_array_equal(control['preprocess'][key],pr[key])
    assert control['preprocess']['cost_scale']==pr['cost_scale']
    torch.manual_seed(seed); model=ContinuationHead(initial,arm)
    initial_state={k:v.detach().clone() for k,v in initial.state_dict().items()}
    optimizer=torch.optim.AdamW(model.learned.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(x,pr)); d=torch.tensor(env/pr['cost_scale'],dtype=torch.float32)
    target=torch.tensor(np.where(known[:,None],y/pr['cost_scale'],0),dtype=torch.float32)
    rms=np.sqrt(np.sum(pr['weights'][:,None]*target.numpy().astype(float)**2,axis=0)).clip(1e-4)
    np.testing.assert_array_equal(rms,control['loss_scales']); scales=torch.tensor(rms,dtype=torch.float32)
    fixed=control['fixed_ids']; assert known[fixed].all()
    groups=[np.flatnonzero(known & (sites==s)) for s in sorted(set(sites))]
    rng=torch.Generator().set_state(control['sampler_rng'])
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step=0; seconds=0.; trace=[]; draws=np.zeros(len(x),np.int64)
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in dict(arm=arm,seed=seed,settings=settings,identity=identity).items(): assert saved[k]==v
        for k,v in initial_state.items(): assert torch.equal(saved['initial_model'][k],v)
        for k in ('mean','std','weights','known'):
            np.testing.assert_array_equal(saved['preprocess'][k],pr[k])
        np.testing.assert_array_equal(saved['loss_scales'],rms)
        np.testing.assert_array_equal(saved['fixed_ids'],fixed)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step,seconds,trace,draws=saved['step'],saved['seconds'],saved['trace'],saved['draws']
    limit=settings['steps'] if stop_at is None else min(settings['steps'],stop_at)
    if limit<step or limit<=0: raise ValueError('Nondecreasing budget required')
    first=step; started=time.monotonic()
    def diagnostic():
        with torch.no_grad(): _,parts=loss_parts(model(z[fixed],d[fixed]),target[fixed],scales)
        return dict(step=step,**parts)
    if not trace:
        with torch.no_grad():
            assert torch.equal(model(z[fixed],d[fixed]),initial(z[fixed],d[fixed]))
        trace.append(diagnostic())
        assert trace[0]['moment_mse']==control['trace'][-1]['moment_mse']
    def save():
        if model.reference is not None:
            for k,v in model.reference.state_dict().items(): assert torch.equal(v,initial_state[k])
        value=dict(arm=arm,seed=seed,settings=settings,identity=identity,initial_model=initial_state,
            preprocess=pr,loss_scales=rms,fixed_ids=fixed,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),step=step,
            seconds=seconds+time.monotonic()-started,trace=trace,draws=draws)
        temp=path.with_suffix('.tmp'); torch.save(value,temp); os.replace(temp,path)
    model.train()
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng); assert known[ids].all()
            optimizer.zero_grad(set_to_none=True)
            loss,_=loss_parts(model(z[ids],d[ids]),target[ids],scales)
            loss.backward(); grad=nn.utils.clip_grad_norm_(model.learned.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); step+=1; np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(diagnostic(),batch_loss=float(loss.detach()),gradient_norm=float(grad)); trace.append(row)
                heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint',step=step); raise
    model.eval()
    return model,dict(step=step,new_updates=step-first,complete=step==settings['steps'],
        seconds=seconds+time.monotonic()-started,trace=trace,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        total_draws=int(draws.sum()),unknown_training_draws=int(draws[~known].sum()),
        reference_frozen=arm=='protected',optimizer_reset=True)


def restore(directory):
    value=torch.load(Path(directory)/'checkpoint.pt',map_location='cpu',weights_only=False)
    model=ContinuationHead(EventMomentHead(len(value['preprocess']['mean']),value['settings']['width']),value['arm'])
    model.load_state_dict(value['model']); model.eval()
    return model,value
