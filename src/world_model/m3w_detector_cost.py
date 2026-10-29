"""Matched causal detector-history cost branch, with exact disabled control."""
from contextlib import contextmanager
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_completion_cost as parent
from src.world_model import m3w_scene_context_cost as scene
from src.world_model.m3w_detector_history_features import noise_features,REPRESENTATIONS

OriginalHead=scene.SceneCostHead


def representation(history,recordings,agents,frames,arm):
    if history.ndim!=2 or history.shape[1]!=64 or not np.isfinite(history).all() or arm not in REPRESENTATIONS:
        raise ValueError('Fixed64-field causal histories and registered representation required')
    if arm=='history':return history.copy()
    if arm=='disabled':return np.zeros_like(history)
    if arm=='noise':return noise_features(recordings,agents,frames)
    out=np.zeros_like(history)
    for group in range(4):out[:,8*group:8*(group+1)]=history[:,8*group+7,None]
    out[:,57:64]=((history[:,7]>0)&(history[:,15]>0))[:,None]
    return out


def normalize(extra,sites,causal):
    if (extra.shape!=(len(sites),64) or causal.shape!=(len(sites),) or causal.dtype!=bool
            or not np.isfinite(extra).all()):
        raise ValueError('Causal fitting feature rows only; no outcome-support argument')
    means=[];seconds=[];names=sorted(set(sites))
    for name in names:
        x=extra[(sites==name)&causal].astype(float)
        if not len(x):raise ValueError('Every fitting source needs causal input support')
        means.append(x.mean(0));seconds.append(np.square(x).mean(0))
    mean=np.mean(means,0);std=np.sqrt(np.maximum(np.mean(seconds,0)-mean**2,0)).clip(1e-6)
    return dict(mean=mean,std=std,clip=8.,fitting_sources=names,future_label_support_used=False)


class DetectorFeatures:
    def __init__(self,base,extra):
        if extra.shape!=(base.shape[0],64) or not torch.isfinite(extra).all():
            raise ValueError('Aligned normalized past detector vectors required')
        self.base,self.extra=base,extra;self.shape=(*base.shape[:2],base.shape[-1]+64)

    def __getitem__(self,rows):
        x=self.base[rows];v=self.extra[rows]
        if x.ndim!=3:raise ValueError('Explicit row batches required')
        return torch.cat((x,v[:,None].expand(-1,x.shape[1],-1)),-1)


class DetectorCostHead(OriginalHead):
    def __init__(self,dimension,width,seed,bias,*,representation='history'):
        if representation not in REPRESENTATIONS or dimension!=462:
            raise ValueError('398 original fields plus64 detector fields required')
        super().__init__(dimension-64,width,seed,bias)
        self.representation=representation
        self.detector_weight=nn.Parameter(torch.zeros(width,64))

    def forward(self,x,envelope,action_ids):
        if x.shape[-1]!=462 or envelope.shape!=x.shape[:2]:
            raise ValueError('Aligned causal candidate/detector features required')
        original=x[...,:398]
        if self.representation=='disabled':return super().forward(original,envelope,action_ids)
        code=F.one_hot(action_ids,7).to(x.dtype).expand(len(x),-1,-1)
        hidden=self.network[0](torch.cat((original[...,:380],code),-1))
        hidden=hidden+F.linear(original[...,380:],self.scene_weight)+F.linear(x[...,398:],self.detector_weight)
        raw=self.network[2](self.network[1](hidden))
        fractions=torch.softmax(torch.cat((raw[...,:2],torch.zeros_like(raw[...,:1])),-1),-1)
        benefit,harm=(envelope[...,None]*fractions[...,:2]).unbind(-1)
        reference=F.softplus(raw[...,2]);benefit=torch.minimum(benefit,reference)
        return torch.stack((benefit,harm,reference,reference*torch.sigmoid(raw[...,3]),harm*torch.sigmoid(raw[...,4])),-1)


@contextmanager
def installed(arm):
    original=scene.SceneCostHead
    scene.SceneCostHead=lambda *a,**kw:DetectorCostHead(*a,representation=arm,**kw)
    try:yield
    finally:scene.SceneCostHead=original


def training_data(original,extra,sites,arm):
    pr=normalize(extra,sites,original['allowed'].numpy().any(1))
    z=np.clip((extra-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32)
    data=dict(original,preprocess=dict(original['preprocess'],detector_normalizer=pr,detector_representation=arm))
    data['z']=DetectorFeatures(original['z'],torch.from_numpy(z))
    data['input_hashes']=dict(original['input_hashes'],causal_detector_features=hashlib.sha256(extra.tobytes()).hexdigest())
    return data


def assert_neutral(old,new):
    if new['preprocess']['detector_representation']!='disabled':raise ValueError('Disabled branch required')
    for key in ('step','trace','draw_hash','row_draws','sampler_rng','torch_rng','mode','arm'):
        parent.base.exact(old[key],new[key])
    for key,value in old['preprocess'].items():parent.base.exact(value,new['preprocess'][key])
    for key,value in old['input_hashes'].items():parent.base.exact(value,new['input_hashes'][key])
    for key,value in old['model'].items():
        parent.base.exact(value,new['model'][key]);parent.base.exact(old['initial_model'][key],new['initial_model'][key])
    if torch.count_nonzero(new['model']['detector_weight']):raise ValueError('Disabled detector weights changed')
    # Direct parameters precede network parameters: the inserted detector weight has no gradient/state.
    mapping={0:0,1:2,2:3,3:4,4:5}
    if set(new['optimizer']['state'])!={mapping[k] for k in old['optimizer']['state']}:
        raise ValueError('Unexpected disabled optimizer state')
    for key,value in old['optimizer']['state'].items():parent.base.exact(value,new['optimizer']['state'][mapping[key]])


def predict(state,x,static,extra,envelope,scales,*,chunk=1024):
    if (x.shape[1:]!=(7,380) or static.shape!=(len(x),7,18) or extra.shape!=(len(x),64)
            or envelope.shape!=(len(x),7) or scales.shape!=(len(x),)
            or not all(np.isfinite(v).all() for v in (x,static,extra,envelope,scales)) or (scales<=0).any() or (envelope<0).any()):
        raise ValueError('Causal original/static/detector inputs and fitting-only scales required')
    pr=state['preprocess'];norm=pr['detector_normalizer']
    model=DetectorCostHead(462,state['settings']['width'],state['seed'],pr['bias'],representation=pr['detector_representation'])
    model.load_state_dict(state['model']);model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),chunk):
            sl=slice(start,start+chunk);base=np.concatenate((x[sl],static[sl]),-1)
            z=np.clip((base-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32)
            e=np.clip((extra[sl]-norm['mean'])/norm['std'],-norm['clip'],norm['clip']).astype(np.float32)
            inputs=np.concatenate((z,np.repeat(e[:,None],7,axis=1)),-1)
            p=model(torch.from_numpy(inputs),torch.from_numpy((envelope[sl]/scales[sl,None]).astype(np.float32)),torch.arange(7))
            result.append(p.numpy()*scales[sl,None,None])
    return np.concatenate(result)
