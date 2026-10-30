"""Matched past detector tokens in the existing bounded SOURCE Transformer."""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_european_source_forecast import SourceForecaster,baseline_torch
from src.world_model.m3w_supervised_intervention import INPUT_KEYS
from src.world_model.m3w_context_conditioning import condition_inputs
from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget,radial_squash
from src.world_model import m3w_native_forecast as native

REPRESENTATIONS=('disabled','current','noise','history')
ORIGINAL_PACK=native.pack_geometry


def pack_geometry(geometry):
    g=np.asarray(geometry,np.float32)
    if g.ndim!=2 or g.shape[1]!=540 or not np.isfinite(g).all():raise ValueError('476 original and64 past detector fields required')
    inputs=ORIGINAL_PACK(g[:,:476])
    inputs['detector_features']=torch.from_numpy(g[:,476:].reshape(-1,8,8).transpose(0,2,1).copy())
    return inputs


class DetectorGeometry:
    def __init__(self,geometry,extra,mean,std,clip=8.):
        if geometry.shape!=(len(extra),476) or extra.shape!=(len(extra),64):raise ValueError('All original producer rows required')
        self.geometry,self.extra,self.mean,self.std,self.clip=geometry,extra,mean,std,clip
        self.shape=(len(extra),540)

    def __getitem__(self,ids):
        x=np.asarray(self.geometry[ids]);det=np.clip((self.extra[ids]-self.mean)/self.std,-self.clip,self.clip).astype(np.float32)
        if x.ndim!=2:raise ValueError('Explicit producer batch row IDs required')
        return np.concatenate((x,det),-1)


def temporal_forward(core,inputs,detector,weight):
    history,neighbors=inputs['history'],inputs['neighbors'].flatten(1,2)
    hmask,nmask=inputs['history_mask'],inputs['neighbor_mask'].flatten(1,2)
    mask=torch.cat((hmask,nmask),1);values=torch.cat((history,neighbors),1)
    if (not hmask.all() or not torch.isfinite(values[mask]).all() or (values[...,2][mask]>0).any()
            or detector.shape!=(len(history),8,8) or not torch.isfinite(detector).all()):
        raise ValueError('Complete finite past detector and trajectory tokens required')
    if core.neighbor_policy!='complete_aligned_history':raise ValueError('Keep original neighbor support policy')
    aligned=torch.isclose(inputs['neighbors'][...,2],history[:,None,:,2],atol=1e-6,rtol=1e-6).all(-1)
    eligible=inputs['neighbor_mask'].all(-1)&aligned
    nmask=(inputs['neighbor_mask']&eligible[...,None]).flatten(1,2);mask=torch.cat((hmask,nmask),1)
    values=torch.where(mask[...,None],values,0.)
    token=core.embed(values)
    token[:,:history.shape[1]]=token[:,:history.shape[1]]+core.modality.weight[0]
    token[:,history.shape[1]:]=token[:,history.shape[1]:]+core.modality.weight[1]
    token=torch.cat((token[:,:8]+F.linear(detector,weight),token[:,8:]),1)
    memory=core.encoder(token,src_key_padding_mask=~mask)
    request=inputs['request_mask'];q=torch.cat((inputs['baseline'],inputs['prediction_time'][...,None]),-1)
    if not torch.isfinite(q[request]).all() or (inputs['prediction_time'][request]<=0).any():raise ValueError('Causal rollout/request required')
    queries=core.query(torch.where(request[...,None],q,0.))
    decoded,_=core.attention(queries,memory,memory,key_padding_mask=~mask,need_weights=False)
    return torch.where(request[...,None],core.output(decoded+queries),0.)


class DetectorForecaster(SourceForecaster):
    def __init__(self,architecture,baseline_index,representation):
        if (representation not in REPRESENTATIONS or architecture.get('output_parameterization')!='motion_bounded'
                or architecture.get('input_conditioning')!='observed_joint_max_norm'):
            raise ValueError('Original bounded/conditioned producer and fixed detector representation required')
        super().__init__(architecture,baseline_index)
        self.representation=representation
        self.detector_weight=nn.Parameter(torch.zeros(architecture['width'],8))

    def forward(self,inputs):
        if set(inputs)!=(INPUT_KEYS|{'detector_features'}):raise ValueError('Only declared past input fields required')
        plain={k:v for k,v in inputs.items() if k!='detector_features'}
        if self.representation=='disabled':return super().forward(plain)
        plain=dict(plain,baseline=baseline_torch(plain['history'],self.baseline_index))
        conditioned,scale=condition_inputs(plain)
        core=self.predictor.predictor.predictor
        delta=temporal_forward(core,conditioned,inputs['detector_features'],self.detector_weight)*scale[:,None,None]
        delta=past_motion_budget(plain)[...,None]*radial_squash(delta)
        request=plain['request_mask'];baseline=torch.where(request[...,None],plain['baseline'],0.)
        return torch.where(request[...,None],baseline+delta,0.)


@contextmanager
def installed():
    original=native.pack_geometry;native.pack_geometry=pack_geometry
    try:yield
    finally:native.pack_geometry=original


def assert_neutral(old,new):
    from src.world_model.m3w_multiaction_cost import exact
    for key in ('settings','seed','step','losses','draws','train_ids','factors','sampler_rng','torch_rng'):
        exact(old[key],new[key])
    for name,value in old['model'].items():exact(value,new['model'][name])
    if torch.count_nonzero(new['model']['detector_weight']):raise ValueError('Disabled branch changed')
    if set(new['optimizer']['state'])!={i+1 for i in old['optimizer']['state']}:raise ValueError('Disabled AdamW state mismatch')
    for i,value in old['optimizer']['state'].items():exact(value,new['optimizer']['state'][i+1])


def assert_recovery(left,right):
    from src.world_model.m3w_multiaction_cost import exact
    for key in left:
        if key!='seconds':exact(left[key],right[key])
