"""Exact-direction diagnostics for shrinking fixed OLS4 replacements."""
import numpy as np


def components(floor, candidate, target, valid):
    """The lower gain is concave in amplitude; return safe masked terms."""
    delta = candidate-floor
    distance = np.linalg.norm(delta,axis=2)
    residual = floor-np.where(valid[...,None],target,0.)
    residual = np.where(valid[...,None],residual,0.)
    count = valid.sum(1)
    return residual, delta, distance, valid, np.maximum(count,1), np.where(count > 0,0.,distance.max(1))


def value_slope(c, amplitude):
    residual,delta,dist,valid,count,debit = c
    a = np.broadcast_to(np.asarray(amplitude),count.shape)
    shifted = residual+a[:,None,None]*delta
    error = np.linalg.norm(shifted,axis=2)
    before = np.linalg.norm(residual,axis=2)
    dot = (shifted*delta).sum(2)
    d = np.divide(dot,error,out=dist.copy(),where=error > 0.)
    value = np.where(valid,before-error,0.).sum(1)/count-a*debit
    slope = -np.where(valid,d,0.).sum(1)/count-debit
    return value,slope


def maximize(evaluate):
    """Maximize scalar or separable concave curves on [0,1] by right slopes."""
    v0,s0 = evaluate(0.)
    v1,s1 = evaluate(1.)
    lo,hi = np.zeros_like(s0),np.ones_like(s0)
    for _ in range(32):
        mid = (lo+hi)*.5
        _,s = evaluate(mid)
        lo,hi = np.where(s > 0.,mid,lo),np.where(s <= 0.,mid,hi)
    t = np.where(s0 <= 0.,0.,np.where(s1 >= 0.,1.,(lo+hi)*.5))
    v,_ = evaluate(t)
    if np.any(v+1e-8 < np.maximum(v0,v1)) or not np.isfinite(v).all():
        raise ValueError('Concave amplitude optimizer failed to dominate endpoints')
    return t,v,s0
