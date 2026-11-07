"""Learn the additional utility and harm of enlarging the same intervention."""
from itertools import permutations

import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors

LEVELS = (.25, .5, 1.)


def outcomes(paths, floor, target, valid, easy_cut):
    """Three absolute terms for fitting and score only; no label in selection."""
    ref,_ = errors(floor,target,valid)
    known = valid.any(1)
    easy = known & (ref>0) & (ref<=easy_cut)
    result = []
    for a in range(paths.shape[1]):
        error,_ = errors(paths[:,a],target,valid)
        radius = np.linalg.norm(paths[:,a]-floor,axis=-1).max(1)
        gain = np.where(known,ref-error,-radius)
        harm = np.where(known,np.maximum(-gain,0.),radius)
        result.append(np.column_stack((gain,harm,np.where(easy|~known,harm,0.))))
    return np.stack(result,axis=1)


def increment(lower, upper):
    d = upper-lower
    d[...,1:] = np.maximum(d[...,1:],0.)
    return d


def fit_moments(x, delta, scale, weights):
    w = weights/weights.sum()
    y = delta/scale[:,None,None]
    return (np.einsum('nad,n,nae->ade',x,w,x,optimize=True),
            np.einsum('nad,n,nac->adc',x,w,y,optimize=True))


def project(delta, radius):
    gain = np.clip(delta[...,0],-radius,radius)
    harm = np.clip(delta[...,1],0.,radius)
    easy = np.clip(delta[...,2],0.,harm)
    return np.stack((gain,harm,easy),axis=-1)


def scores(delta):
    """Final-level score relative to the quarter path."""
    x = delta[...,0]-delta[...,1]-delta[...,2]
    return np.concatenate((np.zeros_like(x[...,:1]),np.cumsum(x,axis=-1)),axis=-1)


def choose(upper, values):
    return np.where(np.arange(3)<=upper[:,None],values,-np.inf).argmax(1)


def match(groups, upper, assigned, values):
    """Independent ranking with the same level multiset at each current query."""
    result = assigned.copy()
    exchangeable = 0
    for g in groups:
        if len(g)>2:
            raise ValueError('The registered allocator selects at most two rows')
        options = sorted(set(permutations(assigned[g])))
        legal = [np.asarray(a,dtype=int) for a in options if np.all(a<=upper[g])]
        if not legal:
            raise ValueError('The original assigned levels must be feasible')
        exchangeable += len(legal)>1
        result[g] = max(legal,key=lambda z:float(values[g,z].sum()))
    return result,exchangeable
