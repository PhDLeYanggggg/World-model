"""Causal zero-step projection of references and benefit,not trajectory outputs."""
import numpy as np
from src.world_model.m3w_frame_action_allocation import validate


def project(prediction,allowed):
    p,ok=validate(prediction,allowed)
    if (p[:,:,4][ok]>p[:,:,1][ok]).any():
        raise ValueError('Inherited easy harm must already be bounded by harm')
    counts=ok.sum(1);out=p.copy();shared={}
    # Center the mean so already-shared references are bitwise fixed points.
    for k in (2,3):
        anchor=p[np.arange(len(p)),ok.argmax(1),k]
        delta=np.where(ok,p[:,:,k]-anchor[:,None],0.).sum(1)
        shared[k]=anchor+np.divide(delta,counts,out=np.zeros(len(p)),where=counts>0)
    shared[3]=np.minimum(shared[3],shared[2])
    for k in (2,3):out[:,:,k][ok]=np.broadcast_to(shared[k][:,None],ok.shape)[ok]
    out[:,:,0][ok]=np.minimum(p[:,:,0][ok],out[:,:,2][ok])
    if not np.isfinite(out).all():raise FloatingPointError('Projection must retain finite native costs')
    if (out[:,:,0][ok]>p[:,:,0][ok]).any():raise AssertionError('Benefit must never be inflated')
    np.testing.assert_array_equal(out[:,:,[1,4]],p[:,:,[1,4]])
    np.testing.assert_array_equal(out[~ok],p[~ok])
    return out
