"""Best-case soft-risk bounds for a fixed, bounded seven-action cost head."""
import numpy as np
from scipy.special import softmax, logsumexp


def minimum_switch_probability(capacity, allowed, temperature):
    cap, ok = np.asarray(capacity, float), np.asarray(allowed)
    if (cap.ndim!=2 or cap.shape[1]!=7 or ok.shape!=cap.shape or ok.dtype!=bool
            or not np.isfinite(cap).all() or (cap<0).any() or not np.isfinite(temperature) or temperature<=0):
        raise ValueError('Finite normalized capacities,causal availability and positive temperature required')
    z=logsumexp(np.where(ok,-cap/temperature,-np.inf),axis=1)
    return np.exp(z-np.logaddexp(0.,z))


def best_case_risk(costs, capacity, allowed, temperature, *, iterations=64):
    """Minimize expected row cost over the superset logit box; not a fitted policy.

    For trial value t,the fractional objective is minimized by choosing the upper
    unnormalized action weight when cost<t,and the lower weight otherwise.
    Root bisection yields the box optimum; ignored moment couplings only relax it.
    """
    cap, ok, y = np.asarray(capacity,float), np.asarray(allowed), np.asarray(costs,float)
    minimum_switch_probability(cap,ok,temperature)
    if y.shape!=(*cap.shape,3) or not np.isfinite(y).all() or iterations<32:
        raise ValueError('Three finite label/bound costs per candidate and adequate precision required')
    low=np.minimum(np.where(ok[...,None],y,np.inf).min(1),0.)
    high=np.maximum(np.where(ok[...,None],y,-np.inf).max(1),0.)
    lower=-cap/temperature; upper=cap/temperature
    for _ in range(iterations):
        mid=(low+high)/2
        z=np.where(y<mid[:,None,:],upper[...,None],lower[...,None])
        z=np.where(ok[...,None],z,-np.inf)
        logits=np.concatenate((np.zeros((len(cap),1,3)),z),1)
        p=softmax(logits,axis=1)[:,1:]
        value=(p*y).sum(1)
        go_up=value>mid
        low=np.where(go_up,mid,low);high=np.where(go_up,high,mid)
    return low
