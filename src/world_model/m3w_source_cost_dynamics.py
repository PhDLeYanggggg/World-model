"""Bounded source-fit trajectory heads with matched ADE/log-ADE objectives."""
import os
import time

import numpy as np
import torch
from torch import nn
from src.world_model.m3w_source_visual_start import VisualStart

OBJECTIVES = ('ade', 'log_ade')


class SourceDynamics(VisualStart):
    def __init__(self, width=480):
        super().__init__(width)
        self.head[-1] = nn.Linear(32, 24)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, geometry, rgb, coverage, arm):
        raw = super().forward(geometry, rgb, coverage, arm).reshape(-1,12,2)
        return raw/(1+torch.linalg.vector_norm(raw,dim=-1,keepdim=True))


def restore(local, radius, rotation, support):
    n=len(local)
    if (local.shape!=(n,12,2) or radius.shape!=(n,) or rotation.shape!=(n,2,2)
            or support.shape!=(n,) or support.dtype!=torch.bool):
        raise ValueError('Aligned twelve-step offsets and observed restoration frame required')
    return torch.where(support[:,None,None],
        torch.einsum('ntd,nkd->ntk',local,rotation)*radius[:,None,None],torch.zeros_like(local))


def dynamics_loss(prediction,target,normalizer,objective):
    if (prediction.ndim!=3 or prediction.shape[1:]!=(12,2) or prediction.shape!=target.shape
            or normalizer<=0 or not np.isfinite(normalizer) or objective not in OBJECTIVES):
        raise ValueError('Fixed objective, positive train-only scale and complete targets required')
    ade=torch.linalg.vector_norm(prediction-target,dim=-1).mean(1)/normalizer
    loss=ade.mean() if objective=='ade' else torch.log1p(ade).mean()
    return loss,ade.mean()


def forecast(model, inputs, ids, arm):
    model.eval(); result=[]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            features,frame=inputs(np.asarray(ids[start:start+128]))
            result.append(restore(model(*features,arm),*frame).numpy())
    return np.concatenate(result)


def fit_dynamics(model,inputs,targets,train_ids,weights,*,arm,objective,normalizer,
                 seed,config,identity,checkpoint,heartbeat,stop_at=None):
    ids=np.asarray(train_ids,dtype=int); w=torch.as_tensor(weights,dtype=torch.float64)
    if (not len(ids) or len(np.unique(ids))!=len(ids) or w.shape!=(len(ids),)
            or torch.any(w<=0) or not torch.isclose(w.sum(),torch.tensor(1.,dtype=w.dtype))):
        raise ValueError('Unique training rows and positive normalized weights required')
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=torch.Generator().manual_seed(seed+7919)
    step,seconds,losses=0,0.,[];draws=np.zeros(len(ids),np.int64)
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if (cp['identity']!=identity or cp['config']!=config or cp['normalizer']!=normalizer
                or cp['arm']!=arm or cp['objective']!=objective):
            raise ValueError('Checkpoint identity/objective/scale changed')
        np.testing.assert_array_equal(cp['train_ids'],ids)
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng']);torch.set_rng_state(cp['torch_rng'])
        step,seconds,losses,draws=cp['step'],cp['fit_seconds'],cp['losses'],cp['draw_counts']
    first=step;started=time.monotonic()
    limit=config['updates'] if stop_at is None else min(stop_at,config['updates'])
    if min(limit,config['batch_size'],config['checkpoint_every'])<=0:
        raise ValueError('Positive training budgets required')
    model.train()
    try:
        while step<limit:
            local=torch.multinomial(w,config['batch_size'],replacement=True,generator=rng).numpy()
            batch=ids[local]; features,frame=inputs(batch); target=targets(batch)
            opt.zero_grad(set_to_none=True)
            prediction=restore(model(*features,arm),*frame)
            loss,ade=dynamics_loss(prediction,target,normalizer,objective)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite trajectory loss')
            loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step();step+=1;np.add.at(draws,local,1)
            if step==1 or step%100==0:
                losses.append(dict(step=step,objective_loss=float(loss.detach()),
                    normalized_batch_ade=float(ade.detach()),gradient_norm=float(grad)))
            if step==limit or step%config['checkpoint_every']==0:
                state=dict(identity=identity,config=config,arm=arm,objective=objective,normalizer=normalizer,
                    model=model.state_dict(),optimizer=opt.state_dict(),sampler_rng=rng.get_state(),
                    torch_rng=torch.get_rng_state(),step=step,fit_seconds=seconds+time.monotonic()-started,
                    losses=losses,train_ids=ids,draw_counts=draws)
                checkpoint.parent.mkdir(parents=True,exist_ok=True);tmp=checkpoint.with_suffix('.tmp')
                torch.save(state,tmp);os.replace(tmp,checkpoint)
                heartbeat(state='training',step=step,objective_loss=float(loss.detach()),
                    normalized_batch_ade=float(ade.detach()),fit_seconds=state['fit_seconds'])
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint',last_completed_step=step)
        raise
    model.eval()
    return dict(step=step,complete=step==config['updates'],new_updates=step-first,
        fit_seconds=seconds+time.monotonic()-started,losses=losses,
        distinct_sampled_rows=int((draws>0).sum()),total_draws=int(draws.sum()))
