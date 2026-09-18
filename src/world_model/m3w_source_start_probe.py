"""Past-only unit-frame start information; proper-score probe, not a policy."""
import os
import time

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def stationary_membership(geometry):
    x=np.asarray(geometry)
    if x.ndim!=2 or x.shape[1]!=476 or not np.isfinite(x).all():
        raise ValueError('Finite past-only 476-column schema required')
    return np.all(x[:,:16]==0,axis=1)


def start_supervision(target,valid):
    target,valid=np.asarray(target),np.asarray(valid)
    if target.shape!=(len(target),12,2) or valid.shape!=target.shape[:2] or valid.dtype!=bool:
        raise ValueError('Twelve-step labels and explicit masks required')
    if not np.isfinite(target[valid]).all():
        raise ValueError('Supported labels must be finite')
    complete=valid.all(1)
    changed=np.any(np.where(valid[...,None],target,0)!=0,axis=(1,2))
    return changed.astype(np.int64),complete


def training_rows(main_x,main_y,source_x,source_y,schedule):
    if schedule=='main_only':
        x,y=main_x,main_y; w=np.full(len(y),1/len(y))
    elif schedule=='source_only':
        x,y=source_x,source_y; w=np.full(len(y),1/len(y))
    elif schedule=='mixed':
        x=np.concatenate((main_x,source_x)); y=np.concatenate((main_y,source_y))
        w=np.r_[np.full(len(main_y),.5/len(main_y)),np.full(len(source_y),.5/len(source_y))]
    else:
        raise ValueError('Fixed source schedule required')
    if not len(y) or not np.isin(y,[0,1]).all() or np.unique(y).size!=2:
        raise ValueError('Both supervised classes required')
    return np.asarray(x),np.asarray(y),w


def fit_normalizer(x,weights):
    x,weights=np.asarray(x,dtype=float),np.asarray(weights,dtype=float)
    if weights.shape!=(len(x),) or np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError('Normalized training-only probability weights required')
    mean=np.sum(x*weights[:,None],axis=0)
    std=np.maximum(np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0)),1e-6)
    return dict(mean=mean,std=std,constant=np.ptp(x,axis=0)==0)


def normalize(x,state):
    z=(np.asarray(x)-state['mean'])/state['std']
    if not np.isfinite(z).all():
        raise ValueError('Finite features required')
    clipped=np.mean(np.abs(z[:,~state['constant']])>10) if (~state['constant']).any() else 0.
    z=np.clip(z,-10,10); z[:,state['constant']]=0
    return z.astype(np.float32),float(clipped)


def metrics(y,p,prior):
    y,p=np.asarray(y),np.asarray(p,dtype=float)
    if y.shape!=p.shape or np.any((p<0)|(p>1)) or not np.isfinite(p).all():
        raise ValueError('Aligned finite probabilities required')
    brier=float(np.mean((p-y)**2)); ref=float(np.mean((prior-y)**2))
    bins=np.minimum((p*10).astype(int),9); curve=[]; ece=0.
    for i in range(10):
        mask=bins==i
        if mask.any():
            row=dict(bin=i,rows=int(mask.sum()),mean_probability=float(p[mask].mean()),rate=float(y[mask].mean()))
            ece+=mask.mean()*abs(row['mean_probability']-row['rate']); curve.append(row)
    q=np.clip(p,1e-7,1-1e-7)
    return dict(rows=len(y),positive_rate=float(y.mean()),brier=brier,reference_brier=ref,
        brier_lift=ref-brier,auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
        log_loss=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),ece=float(ece),calibration_curve=curve,
        probability_not_trajectory_utility=True)


class StartMLP(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)


def fit_mlp(x,y,weights,*,seed,config,identity,checkpoint,heartbeat,stop_at=None):
    torch.manual_seed(seed); model=StartMLP(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(seed+7919)
    step,seconds,losses=0,0.,[]
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['config']!=config:
            raise ValueError('Resume data/model identity changed')
        model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer'])
        generator.set_state(cp['sampler_rng']); torch.set_rng_state(cp['torch_rng'])
        step,seconds,losses=cp['step'],cp['seconds'],cp['losses']
    initial=step; start=time.monotonic(); limit=min(config['updates'],stop_at) if stop_at is not None else config['updates']
    if limit<=0 or config['batch_size']<=0 or config['checkpoint_every']<=0:
        raise ValueError('Positive training budgets required')
    x=torch.as_tensor(x,dtype=torch.float32); y=torch.as_tensor(y,dtype=torch.float32); weights=torch.as_tensor(weights,dtype=torch.float64)
    model.train()
    while step<limit:
        ids=torch.multinomial(weights,config['batch_size'],replacement=True,generator=generator)
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(model(x[ids]),y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite classifier loss')
        loss.backward(); gradient=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); opt.step(); step+=1
        if step==1 or step%100==0:
            losses.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(gradient)))
        if step==limit or step%config['checkpoint_every']==0:
            state=dict(identity=identity,config=config,model=model.state_dict(),optimizer=opt.state_dict(),
                sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),step=step,seconds=seconds+time.monotonic()-start,losses=losses)
            checkpoint.parent.mkdir(parents=True,exist_ok=True); tmp=checkpoint.with_suffix('.tmp')
            torch.save(state,tmp); os.replace(tmp,checkpoint)
            heartbeat(step=step,loss=float(loss.detach()),seconds=state['seconds'])
    model.eval()
    return model,dict(step=step,complete=step==config['updates'],new_updates=step-initial,seconds=seconds+time.monotonic()-start,losses=losses)


def probabilities(model,x):
    if isinstance(model,nn.Module):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.as_tensor(x[i:i+512])).sigmoid().numpy() for i in range(0,len(x),512)])
    return model.predict_proba(x)[:,1]


def fit_trees(x,y,weights,*,seed,config,identity,checkpoint,heartbeat,stop_at=None):
    start=time.monotonic(); prior_seconds=0.
    if checkpoint.exists():
        bundle=joblib.load(checkpoint)
        if bundle['identity']!=identity or bundle['config']!=config:
            raise ValueError('Partial tree identity/config changed')
        model=bundle['model']; prior_seconds=bundle['fit_seconds']
    else:
        model=ExtraTreesClassifier(**dict(config,n_estimators=0),random_state=seed,n_jobs=1,warm_start=True)
    initial=model.n_estimators; limit=min(config['n_estimators'],stop_at) if stop_at is not None else config['n_estimators']
    if limit<1:
        raise ValueError('Positive tree budget required')
    while model.n_estimators<limit:
        model.set_params(n_estimators=min(limit,model.n_estimators+32)); model.fit(x,y,sample_weight=weights)
        state=dict(model=model,identity=identity,config=config,fit_seconds=prior_seconds+time.monotonic()-start)
        checkpoint.parent.mkdir(parents=True,exist_ok=True); tmp=checkpoint.with_suffix('.tmp')
        joblib.dump(state,tmp); os.replace(tmp,checkpoint)
        heartbeat(trees=model.n_estimators)
    return model,dict(seconds=prior_seconds+time.monotonic()-start,complete=model.n_estimators==config['n_estimators'],
        trees=model.n_estimators,new_trees=model.n_estimators-initial,new_updates=0)
