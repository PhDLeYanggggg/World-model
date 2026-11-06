"""Matched complete-cost versus role-separated learning of bound contributions."""
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_neural_risk_transport as storage

ARMS = ('composed_anchor', 'role_separated')
SETTINGS = dict(steps=4096, batch_size=512, learning_rate=.001, weight_decay=.0001,
                gradient_clip=5., checkpoint_steps=128, log_every=256,
                width1=64, width2=32, observation_width=32, underharm_weight=4.)


def prepare(data, source):
    x,y,w,known,env,sites = (np.asarray(data[k]) for k in
                           ('x','moments','weights','observed','envelope','sites'))
    fit=sites!=source; n=len(x)
    if (x.shape!=(n,120) or y.shape!=(n,5) or w.shape!=(n,) or known.shape!=(n,)
            or known.dtype!=bool or env.shape!=(n,) or not fit.any() or fit.all()
            or not all(np.isfinite(v).all() for v in (x,y,w,env)) or (y<0).any()
            or (w<0).any() or (env<0).any() or w[fit].sum()<=0):
        raise ValueError('Finite complete contributions and an excluded query source required')
    ww=w[fit]/w[fit].sum(); observed=known[fit]
    if not observed.any() or observed.all():
        raise ValueError('Both observed and bound-only fitting roles required')
    ko=fit&known; kw=w[ko]/w[ko].sum()
    mean=np.einsum('n,nd->d',kw,x[ko])
    std=np.sqrt(np.einsum('n,nd->d',kw,(x[ko]-mean)**2)).clip(1e-6)
    rms=np.sqrt(np.einsum('n,nd->d',kw,y[ko]**2)).clip(.05).astype(np.float32)
    prior=np.einsum('n,nd->d',kw,y[ko]); radius=float(kw@env[ko])
    leftover=max(radius-prior[0]-prior[1],1e-5)
    fractions=np.maximum([prior[0],prior[1],leftover],1e-5)
    reference=max(float(prior[2]),1e-5)
    gates=np.clip([prior[3]/reference,prior[4]/max(prior[1],1e-5)],1e-5,1-1e-5)
    bias=np.r_[np.log(fractions[:2]/fractions[2]),
                reference+np.log(-np.expm1(-reference)),np.log(gates/(1-gates))].astype(np.float32)
    probability=float(ww@observed)
    binding=dict(mean=mean,std=std,rms=rms,bias=bias,observation_prior=probability,
        train_ids=data['ids'][fit],train_sources=sorted(set(sites[fit])),held_source=source,
        fit_input_hash=storage.digest(x[fit]),fit_moment_hash=storage.digest(y[fit]),
        fit_weight_hash=storage.digest(ww),fit_role_hash=storage.digest(observed),
        fit_envelope_hash=storage.digest(env[fit]),known_only_normalization=True)
    return dict(x=torch.from_numpy(np.clip((x[fit]-mean)/std,-8,8).astype(np.float32)),
        y=torch.from_numpy(y[fit].astype(np.float32)),known=torch.from_numpy(observed),
        envelope=torch.from_numpy(env[fit].astype(np.float32)),
        probability=torch.from_numpy(ww),rms=torch.from_numpy(rms),binding=binding)


class Head(nn.Module):
    def __init__(self,cfg,preprocess):
        super().__init__()
        self.conditional=nn.Sequential(nn.Linear(120,cfg['width1']),nn.SiLU(),
            nn.Linear(cfg['width1'],cfg['width2']),nn.SiLU(),nn.Linear(cfg['width2'],5))
        # First50 are agent/neighbor motion only, not action-rollout descriptors.
        self.observation=nn.Sequential(nn.Linear(50,cfg['observation_width']),nn.SiLU(),
            nn.Linear(cfg['observation_width'],1))
        nn.init.zeros_(self.conditional[-1].weight)
        nn.init.zeros_(self.observation[-1].weight)
        with torch.no_grad():
            self.conditional[-1].bias.copy_(torch.as_tensor(preprocess['bias']))
            q=preprocess['observation_prior']
            self.observation[-1].bias.fill_(np.log(q/(1-q)))

    def forward(self,x,envelope):
        if x.shape!=(len(envelope),120) or envelope.ndim!=1:
            raise ValueError('Causal120 and causal disagreement only; no role/mask input')
        raw=self.conditional(x)
        fraction=torch.softmax(torch.cat((raw[:,:2],torch.zeros_like(raw[:,:1])),1),1)
        reference=F.softplus(raw[:,2]);harm=envelope*fraction[:,1]
        conditional=torch.stack((torch.minimum(envelope*fraction[:,0],reference),harm,
            reference,reference*torch.sigmoid(raw[:,3]),harm*torch.sigmoid(raw[:,4])),1)
        logit=self.observation(x[:,:50])[:,0];q=torch.sigmoid(logit)
        unknown=torch.stack((torch.zeros_like(envelope),envelope,torch.zeros_like(envelope),
                             torch.zeros_like(envelope),envelope),1)
        return dict(conditional=conditional,observation_logit=logit,observation_probability=q,
                    moments=q[:,None]*conditional+(1-q[:,None])*unknown)


def objective(parts,y,known,rms,arm):
    if arm not in ARMS or y.shape!=(len(known),5) or known.dtype!=torch.bool:
        raise ValueError('Declared arm and training-only observed-role labels required')
    def error(prediction):
        weight=torch.ones_like(prediction)
        for k in (1,4):weight[:,k]=torch.where(prediction[:,k]<y[:,k],SETTINGS['underharm_weight'],1.)
        return (((prediction-y)/rms)**2*weight).mean(1)
    composed=error(parts['moments']).mean()
    conditional=(error(parts['conditional'])*known).sum()/known.sum().clamp_min(1)
    bce=F.binary_cross_entropy_with_logits(parts['observation_logit'],known.float())
    total=composed if arm=='composed_anchor' else conditional+bce
    return dict(total=total,composed_MSE=composed,observed_conditional_MSE=conditional,observation_BCE=bce)


def fit(data,source,identity,path,*,arm,seed,resume=False,stop_at=None,
        heartbeat=lambda **_:None,guard=lambda:None):
    if arm not in ARMS or seed not in (17,29,43):raise ValueError('Registered arm and risk seed required')
    p=prepare(data,source);cfg=SETTINGS
    torch.manual_seed(seed);model=Head(cfg,p['binding']);initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=torch.Generator().manual_seed(seed);prob=p['probability'];n=len(prob)
    binding=dict(identity=identity,settings=cfg,arm=arm,seed=seed,preprocess=p['binding'],
                 initial_model=initial,parameters=sum(v.numel() for v in model.parameters()),fit_rows=n)
    step,elapsed,trace,visits=0,0.,[],np.zeros(n,np.int32);path=Path(path)
    if path.exists():
        if not resume:raise ValueError('Explicit unchanged-definition resume required')
        saved=storage.storage.base.read_checkpoint(path)
        for k,v in binding.items():storage.storage.base.exact(saved[k],v)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['generator']);torch.set_rng_state(saved['torch_rng'])
        step,elapsed,trace,visits=(saved[k] for k in ('step','seconds','trace','visits'))
    limit=cfg['steps'] if stop_at is None else stop_at
    if not step<=limit<=cfg['steps']:raise ValueError('Fixed nondecreasing update budget required')
    start=time.monotonic()
    def save():
        path.parent.mkdir(parents=True,exist_ok=True)
        storage.storage.base.save_checkpoint(path,dict(**binding,model=model.state_dict(),optimizer=opt.state_dict(),
            generator=rng.get_state(),torch_rng=torch.get_rng_state(),step=step,
            seconds=elapsed+time.monotonic()-start,trace=trace,visits=visits,complete=step==cfg['steps']))
    try:
        while step<limit:
            guard();ix=torch.multinomial(prob,cfg['batch_size'],replacement=True,generator=rng)
            opt.zero_grad(set_to_none=True)
            terms=objective(model(p['x'][ix],p['envelope'][ix]),p['y'][ix],p['known'][ix],p['rms'],arm)
            if not torch.isfinite(terms['total']):raise FloatingPointError('Nonfinite role-separated loss')
            terms['total'].backward();norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            opt.step();step+=1;np.add.at(visits,ix.numpy(),1)
            if step==1 or step%cfg['log_every']==0:
                row=dict(step=step,gradient_norm=float(norm),**{k:float(v.detach()) for k,v in terms.items()},
                         distinct_fitting_rows_drawn=int((visits>0).sum()))
                trace.append(row);heartbeat(state='role_cost_training',arm=arm,**row)
            if step%cfg['checkpoint_steps']==0 or step==limit:save()
    except BaseException:
        save();raise
    return storage.storage.base.read_checkpoint(path)


def predict(state,x,envelope):
    x,env=np.asarray(x),np.asarray(envelope)
    if x.shape!=(len(env),120) or not np.isfinite(x).all() or not np.isfinite(env).all() or (env<0).any():
        raise ValueError('Finite causal120 and past-defined candidate envelope required')
    pr=state['preprocess'];model=Head(state['settings'],pr);model.load_state_dict(state['model']);model.eval()
    z=np.clip((x-pr['mean'])/pr['std'],-8,8).astype(np.float32)
    output={k:[] for k in ('conditional','moments','observation_probability')}
    with torch.no_grad():
        for start in range(0,len(x),512):
            out=model(torch.from_numpy(z[start:start+512]),torch.from_numpy(env[start:start+512].astype(np.float32)))
            for k in output:output[k].append(out[k].numpy())
    return {k:np.concatenate(v).astype(float) for k,v in output.items()}


def exact(a,b):
    for k in ('identity','settings','arm','seed','preprocess','initial_model','model','optimizer',
              'generator','torch_rng','step','trace','visits','complete','fit_rows'):
        storage.storage.base.exact(a[k],b[k])
