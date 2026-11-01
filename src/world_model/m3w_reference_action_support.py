"""Causal reference availability without the old predicted-gain sign restriction."""
import numpy as np
from src.world_model import m3w_frame_action_allocation as original


def reference_actions(prediction,allowed):
    p,ok=original.validate(prediction,allowed)
    return ok&(p[:,:,2]>0)&(p[:,:,3]>0)


def pointwise_actions(prediction,allowed):
    p,ok=original.validate(prediction,allowed)
    return reference_actions(p,ok)&(p[:,:,1]<=.02*p[:,:,2])&(p[:,:,4]<=.02*p[:,:,3])


def feasible(prediction,allowed,choices):
    p,ok=original.validate(prediction,allowed);c=np.asarray(choices)
    if c.shape!=(len(p),) or not np.issubdtype(c.dtype,np.integer) or ((c < -1)|(c >= 7)).any():return False
    at=np.flatnonzero(c>=0)
    return bool(reference_actions(p,ok)[at,c[at]].all() and max(original.terms(p,c)[1])<=0)
