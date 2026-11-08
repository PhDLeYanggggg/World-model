"""Expected gain and harm for a small, past-only OLS4 replacement menu."""
import numpy as np
import torch
from torch import nn


NAMES = ('ols3', 'ols8', 'damped4', 'bounded_acceleration4')
ARMS = ('cv_hindcast', 'ols_motion', 'ols_hindcast', 'hindcast_rule')
HINDCAST_WIDTH = 4


def velocity(history, n):
    tail = history[:, -n:]
    t = np.arange(tail.shape[1], dtype=float)
    t -= t.mean()
    return ((tail-tail[:, -1:])*t[None,:,None]).sum(1)/(t@t)


def rollout(history, steps):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1] < 4 or h.shape[2] != 2 or not np.isfinite(h).all():
        raise ValueError('Require a complete causal history of four or more 2D points')
    t = np.arange(1,steps+1,dtype=float)
    v = velocity(h,4)
    cv = np.diff(h[:,-2:],axis=1)*t[None,:,None]
    ols = v[:,None]*t[None,:,None]
    three = velocity(h,3)[:,None]*t[None,:,None]
    eight = velocity(h,8)[:,None]*t[None,:,None]
    damp = v[:,None]*np.cumsum(.90**(t-1))[None,:,None]
    dv = h[:,1:]-h[:,:-1]
    scale = np.sqrt(np.square(dv).sum(2).mean(1)).clip(.05)
    a = np.diff(dv[:,-3:],axis=1).mean(1)
    limit = .15*scale/np.linalg.norm(a,axis=1).clip(1e-8)
    a *= np.minimum(1.,limit)[:,None]
    acceler = ols+a[:,None]*np.minimum(t*(t+1)/2.,2.*t)[None,:,None]
    return cv,ols,np.stack((three,eight,damp,acceler),axis=1),scale


def features(history):
    h = np.asarray(history,dtype=float)
    if h.shape[1:] != (8,2):
        raise ValueError('Expected the registered eight past positions')
    cv,ols,bank,scale = rollout(h,12)
    c, o = [np.linalg.norm(bank-ref[:,None],axis=-1)/scale[:,None,None] for ref in (cv,ols)]
    vel = np.diff(h,axis=1)/scale[:,None,None]
    ds = np.linalg.norm(vel,axis=2)
    jerk = np.linalg.norm(np.diff(vel,n=2,axis=1),axis=2)
    cosine = (vel[:,-1]*vel[:,-2]).sum(1)/(ds[:,-1]*ds[:,-2]).clip(1e-8)
    motion = np.column_stack((np.log1p(scale),ds[:,-1],ds[:,-3:].mean(1),ds.std(1),
                              jerk[:,-1],jerk.mean(1),cosine,
                              np.linalg.norm(cv[:,-1]-ols[:,-1],axis=1)/scale))
    broadcast = np.broadcast_to(motion[:,None],(*bank.shape[:2],motion.shape[1]))
    eye = np.broadcast_to(np.eye(len(NAMES))[None],(*bank.shape[:2],len(NAMES)))
    geometry = np.stack((c.mean(2),c[:,:,-1],o.mean(2),o[:,:,-1]),axis=2)
    probes = []
    for prefix in (4,5,6):
        c0,o0,b0,s = rollout(h[:,:prefix],8-prefix)
        y = h[:,prefix:]-h[:,prefix-1:prefix]
        err = np.linalg.norm(b0-y[:,None],axis=3).mean(2)
        g = (np.linalg.norm(o0-y,axis=2).mean(1)[:,None]-err)/s[:,None]
        q = (np.linalg.norm(c0-y,axis=2).mean(1)[:,None]-err)/s[:,None]
        probes.append((g,q))
    g,q = (np.stack(a,axis=2) for a in zip(*probes,strict=True))
    hind = np.stack((g.mean(2),q.mean(2),np.maximum(-g,0.).mean(2),(g > 0.).mean(2)),axis=2)
    x = np.concatenate((broadcast,eye,geometry,hind),axis=2).clip(-16.,16.).astype(np.float32)
    return x,cv,ols,bank,scale


def targets(ref, bank, future, valid, scale):
    """Future inputs are training labels only; unavailable rows pay disagreement."""
    clean = np.where(valid[...,None],future,0.)
    count = valid.sum(1)
    e = np.linalg.norm(bank-clean[:,None],axis=-1)
    baseline = np.linalg.norm(ref-clean,axis=-1)
    gain = np.where(valid[:,None],baseline[:,None]-e,0.).sum(2)/np.maximum(count[:,None],1)
    bound = np.linalg.norm(bank-ref[:,None],axis=-1).max(2)
    gain = np.where(count[:,None] > 0,gain,-bound)/(12.*scale[:,None])
    return np.stack((gain,np.maximum(-gain,0.)),axis=-1).clip(-4.,4.).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2))


def prediction(net,x):
    value = net(x)
    return torch.stack((value[...,0],torch.nn.functional.softplus(value[...,1])),dim=-1)


def choose(scores, legal, groups):
    scores = np.asarray(scores)
    if scores.shape != legal.shape or not np.isfinite(scores).all():
        raise ValueError('Require aligned finite candidate scores')
    chosen = np.full(len(scores),-1,dtype=np.int16)
    best = np.where(legal,scores,-np.inf).argmax(1)
    for q in groups:
        eligible = q[legal[q].any(1)]
        k = min(2,len(q)-1,len(eligible))
        rank = np.argsort(-scores[eligible,best[eligible]],kind='stable')[:k]
        at = eligible[rank]
        chosen[at] = best[at]
    return chosen
