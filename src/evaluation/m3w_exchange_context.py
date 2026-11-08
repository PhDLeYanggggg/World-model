"""Causal actor-frame interaction features at the existing observation grid."""
import numpy as np


def relative(history, neighbors, rotation, scale):
    """Two neighbor slots, 11 features each; only consecutive observed positions."""
    valid = np.isfinite(neighbors).all(-1)
    current = valid[:,:,-1]
    consecutive = current & valid[:,:,-2]
    delta = np.where(valid[...,None],neighbors-history[:,None],0.)
    local = np.einsum('nkti,nij->nktj',delta,rotation)
    r = local[:,:,-1]
    dv = np.where(consecutive[...,None],r-local[:,:,-2],0.)
    distance = np.linalg.norm(r,axis=-1)
    radial = (r*dv).sum(2)
    tc = np.divide(-radial,np.square(dv).sum(2).clip(1e-10))
    approaching = consecutive & (radial<0)
    tc = np.where(approaching,tc.clip(0.,12.),0.)
    closest = np.linalg.norm(r+tc[...,None]*dv,axis=-1)
    distances = np.linalg.norm(local,axis=-1)
    min_dist = np.where(valid,distances,np.inf).min(2)
    min_dist = np.where(valid.any(2),min_dist,0.)
    s = scale[:,None]
    values = np.stack((current,consecutive,valid.mean(2),
        (r[:,:,0]/s).clip(-20.,20.),(r[:,:,1]/s).clip(-20.,20.),
        np.log1p(distance/s),np.log1p(min_dist/s),
        (-radial/distance.clip(1e-9)/s).clip(-20.,20.),
        np.log1p(np.linalg.norm(dv,axis=-1)/s),tc/12.,np.log1p(closest/s)),axis=2)
    if not np.isfinite(values).all():
        raise ValueError('Unexpected interaction feature')
    return values.reshape(len(history),-1).astype(np.float32)


def shuffle_within_query(values, offsets):
    """A deterministic contemporaneous negative control, never uses future queries."""
    out = np.empty_like(values)
    for lo,hi in zip(offsets[:-1],offsets[1:],strict=True):
        out[lo:hi] = np.roll(values[lo:hi],1,axis=0)
    return out


def feature_arms(own,ped,traffic,offsets):
    zp,zt = np.zeros_like(ped),np.zeros_like(traffic)
    return {k:np.column_stack((own,p,t)) for k,p,t in (
        ('own',zp,zt),('pedestrian',ped,zt),('traffic',ped,traffic),
        ('shuffled_traffic',ped,shuffle_within_query(traffic,offsets)))}
