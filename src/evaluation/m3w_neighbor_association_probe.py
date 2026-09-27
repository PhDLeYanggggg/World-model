"""Observation-only identity-association probe; no trajectory accuracy claim."""
import numpy as np


def scramble_associations(geometry):
    g=np.asarray(geometry,np.float32)
    if g.ndim!=2 or g.shape[1]!=476 or not np.isfinite(g).all():
        raise ValueError('Finite causal geometry required')
    mask=g[:,230:294].reshape(-1,8,8)>0
    if not (mask.all(2).sum(1)>=2).all():
        raise ValueError('Two complete neighbor histories required')
    out=g.copy(); before=g[:,38:166].reshape(-1,8,8,2); after=out[:,38:166].reshape(-1,8,8,2)
    for row,m in enumerate(mask):
        a,b=np.flatnonzero(m.all(1))[:2]
        for t in (1,3,5):
            after[row,a,t]=before[row,b,t]
            after[row,b,t]=before[row,a,t]
    return out


def neighbor_path_length(geometry):
    g=np.asarray(geometry); xy=g[:,38:166].reshape(-1,8,8,2)
    mask=g[:,230:294].reshape(-1,8,8)>0
    use=mask[:,:,1:]&mask[:,:,:-1]
    return np.where(use,np.linalg.norm(np.diff(xy,axis=2),axis=-1),0).sum((1,2))
