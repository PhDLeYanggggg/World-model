"""Past-only scores for a single fixed OLS4 anchor intervention."""
import numpy as np
import torch
from torch import nn

from src.world_model.m3w_ols_anchor_denoising import design
from src.world_model.m3w_ols_replacement import velocity


FRACTION = .5
MARGIN = .1
FITS = ('learned_probe','learned_motion')
ARMS = (*FITS,'prefix_rule','random','oracle')


def past_features(history):
    h = np.asarray(history,dtype=float)
    x,paths,anchor,_,_,_ = design(h)
    probe = []
    for prefix in (4,5,6):
        past = h[:,:prefix]-h[:,prefix-1:prefix]
        v = velocity(past,4)
        bias = past[:,-4:].mean(1)+1.5*v
        t = np.arange(1.,9-prefix)
        ref = v[:,None]*t[None,:,None]
        diff = FRACTION*bias[:,None]*np.minimum(t/3.,1.)[None,:,None]
        actual = h[:,prefix:]-h[:,prefix-1:prefix]
        err = np.linalg.norm(ref-actual,axis=2)-np.linalg.norm(ref+diff-actual,axis=2)
        radius = np.linalg.norm(diff,axis=2).max(1)
        probe.append(np.divide(err.mean(1),radius,out=np.zeros(len(h)),where=radius>1e-8))
    p = np.stack(probe,axis=1)
    z = np.column_stack((p,p.mean(1),(p > 0.).mean(1))).clip(-1.,1.).astype(np.float32)
    return np.column_stack((x,z)),z[:,3],paths['cv'],paths['ols4'],paths['ols4']+FRACTION*anchor


def supervision(floor,change,target,valid):
    y = np.where(valid[...,None],target,0.)
    known = valid.any(1)
    radius = np.linalg.norm(change-floor,axis=2).max(1)
    signed = np.linalg.norm(floor-y,axis=2)-np.linalg.norm(change-y,axis=2)
    gain = np.where(valid,signed,0.).sum(1)/valid.sum(1).clip(1)
    lower = np.where(known,gain,-radius)
    label = np.full(len(radius),1,dtype=np.int64)
    label[lower < -MARGIN*radius] = 0
    label[lower > MARGIN*radius] = 2
    return label,lower,radius > 1e-6,known


def network(width):
    return nn.Sequential(nn.Linear(width,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,3))


def score(net,x,arm):
    if arm not in FITS:
        raise ValueError('Unexpected retrieval arm')
    if arm == 'learned_motion':
        x = x.clone(); x[:,-5:] = 0.
    return net(x)


def choose(values,legal,groups,*,threshold=None):
    if values.shape != legal.shape or not np.isfinite(values).all():
        raise ValueError('Aligned finite ranking scores required')
    out = np.zeros(len(values),dtype=bool)
    for q in groups:
        if len(q) <= 1:
            continue
        at = q[legal[q]]
        if len(at):
            j = at[values[at].argmax()]
            if threshold is None or values[j] > threshold:
                out[j] = True
    return out
