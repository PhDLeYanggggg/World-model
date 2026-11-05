"""Ego-history motion frame with the original context scale retained separately."""
import numpy as np
from src.world_model import m3w_kinematic_risk as kin
from src.world_model import m3w_rollout_history_risk as rel
from src.world_model import m3w_candidate_neighbor_basis as neighbors

ARMS=('history120','history121')
XY=np.r_[np.array([3*i+j for i in range(8) for j in (0,1)]),
    np.array([32+3*i+j for i in range(64) for j in (0,1)]),np.arange(306,354)]


def coordinates(x,scales):
    x,scales=np.asarray(x),np.asarray(scales,float)
    if (x.ndim!=3 or x.shape[1:]!=(7,428) or scales.shape!=(len(x),)
            or not np.isfinite(x).all() or not np.isfinite(scales).all() or (scales<1).any()
            or not (x[...,24:32]==1).all()):
        raise ValueError('Exact causal428 with positive original image-coordinate scales required')
    h=x[:,:,:24].reshape(-1,7,8,3)[...,:2].astype(float)
    if not np.array_equal(h,np.broadcast_to(h[:,:1],h.shape)):
        raise ValueError('Observed history must not change with candidate action')
    radius=np.linalg.norm(h[:,0],axis=-1).max(-1)*scales
    ratio=np.maximum(1.,radius)/scales
    if (ratio>1+1e-5).any():
        raise ValueError('Original scale must cover ego history')
    out=x.copy(); out[...,XY]=(x[...,XY]/ratio[:,None,None]).astype(x.dtype)
    return out,ratio


def summary(x):
    return np.concatenate((kin.motion_basis(x),rel.basis(x),neighbors.basis(x)),axis=-1)


def chunked(x,scales,chunk=512):
    result={k:[] for k in ARMS}; ratios=[]
    for start in range(0,len(x),chunk):
        z,r=coordinates(x[start:start+chunk],scales[start:start+chunk]); base=summary(z)
        result['history120'].append(base)
        result['history121'].append(np.concatenate((base,np.broadcast_to(np.log(r)[:,None,None],(*base.shape[:2],1))),-1))
        ratios.append(r)
    return {k:np.concatenate(v) for k,v in result.items()},np.concatenate(ratios)
