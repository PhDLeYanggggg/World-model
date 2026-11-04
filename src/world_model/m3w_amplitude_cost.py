"""Matched causal amplitude cost heads; future targets are supervision only."""
import numpy as np
import torch
from src.evaluation.m3w_amplitude_opportunity import FRACTIONS,rollout
from src.world_model import m3w_shared_motion_cost as original
from src.world_model.m3w_floor_relative import matched_features
from src.evaluation.m3w_native_metrics import native_errors

ARMS=('binary_only','amplitude_bank')
base=original.base


def causal_bank(*,old_causal,geometry,shared_context,paths,floor_index):
    g,h,p=map(np.asarray,(geometry,shared_context,paths))
    if (set(old_causal) != base.base.CAUSAL_KEYS or g.shape != (len(p),476)
            or h.shape != (len(p),48) or p.shape != (len(p),7,12,2)
            or floor_index not in range(6) or old_causal['x'].shape != (len(p),7,428)
            or not all(np.isfinite(v).all()for v in (g,h,p))):
        raise ValueError('Original exact causal inventory and seven finite saved rollouts required')
    floor,cv=p[:,floor_index],p[:,1]
    paths=np.stack([floor,*[rollout(floor,p[:,6],a)for a in FRACTIONS]],1)
    x=np.empty((len(p),7,428),np.float32)
    switch=np.any(floor != cv,axis=(1,2))
    for k in range(7):
        x[:,k,:380]=matched_features(g,cv,floor,paths[:,k],switch)[0]
        x[:,k,380:]=h
    if not np.array_equal(x[:,6],old_causal['x'][:,6]):
        raise ValueError('Full-amplitude causal fields must equal the original proposal fields')
    envelope=np.linalg.norm(paths-floor[:,None],axis=-1).max(-1)
    allowed=envelope>0
    allowed[:,0]=False
    return dict(paths=paths,cv=cv,floor=floor,
        causal=dict(old_causal,x=x,envelope=envelope,allowed=allowed))


def targets(packet,target,valid,easy_cut):
    paths,c=packet['paths'],packet['causal']
    if not np.isfinite(easy_cut) or easy_cut<=0:
        raise ValueError('Original producer-fitting easy cutoff required')
    errors=np.stack([native_errors(paths[:,k],target,valid,np.ones(len(paths)))[0]for k in range(7)],1)
    cv=native_errors(packet['cv'],target,valid,np.ones(len(paths)))[0]
    reference=errors[:,0]
    known=np.isfinite(reference)
    easy=known & (cv>0) & (cv<=easy_cut)
    benefit=np.maximum(reference[:,None]-errors,0)
    harm=np.maximum(errors-reference[:,None],0)
    r=np.broadcast_to(reference[:,None],errors.shape)
    moments=np.stack((benefit,harm,r,r*easy[:,None],harm*easy[:,None]),-1)
    moments[~known]=0
    moments[~known,:,1]=c['envelope'][~known]
    moments[~known,:,4]=c['envelope'][~known]
    if (not np.isfinite(moments).all() or (moments<0).any()
            or np.any(moments[...,:2].sum(-1)>c['envelope']+1e-6)):
        raise ValueError('Coherent observed moments or unknown bound contributions required')
    return dict(moments=moments,observed_future=known)


def training_data(packet,target,valid,easy_cut,preprocess):
    c,pr=packet['causal'],preprocess
    if (np.shape(pr['mean']) != (428,) or np.shape(pr['std']) != (428,)
            or np.shape(pr['rms']) != (5,) or not np.isfinite(pr['rms']).all()
            or np.any(pr['rms']<=0)):
        raise ValueError('Reuse original physical-fitting normalization without a new fit')
    truth=targets(packet,target,valid,easy_cut)
    z=np.clip((c['x']-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32)
    z[...,380:]=0
    return dict(causal=c,preprocess=pr,z=torch.from_numpy(z),
        truth_by_arm=dict(bound_aware=torch.from_numpy((truth['moments']/c['scales'][:,None,None]).astype(np.float32))),
        defined_by_arm=dict(bound_aware=np.ones(len(z),bool)),observed_future=truth['observed_future'])


def for_arm(data,arm):
    if arm not in ARMS:
        raise ValueError('Registered binary/amplitude comparison required')
    mask=data['causal']['allowed'].copy()
    if arm=='binary_only':
        mask[:,:6]=False
    return dict(data,causal=dict(data['causal'],allowed=mask))


def fit(data,*,arm,**kwargs):
    identity=kwargs.pop('identity')
    return base.fit(for_arm(data,arm),arm='bound_aware',
        identity=dict(identity,feature_arm='local_history',amplitude_arm=arm,
            amplitude_schema='floor_then_six_fixed_fractions_true_CV_separate'),**kwargs)


def predict(state,causal):
    if (state['identity'].get('amplitude_arm') not in ARMS
            or state['identity'].get('amplitude_schema') != 'floor_then_six_fixed_fractions_true_CV_separate'):
        raise ValueError('Registered amplitude schema required, not an inherited binary-bank decoder')
    return original.predict(state,causal)
