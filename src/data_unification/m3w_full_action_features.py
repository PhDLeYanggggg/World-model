"""The original raw120 motion basis for every causal amplitude, not only proposals."""
import numpy as np
from src.data_unification import m3w_complete_policy_packet as packet


def summary(causal):
    x=causal['x']
    result=np.concatenate((packet.relation.parent.kin.motion_basis(x),packet.relation.basis(x),
                           packet.neighbors.chunked(x)),-1).astype(np.float32)
    if result.shape!=(len(x),7,120) or not np.isfinite(result).all():
        raise ValueError('Finite original120 for all seven causal actions required')
    if not np.array_equal(result[:,:,:50],np.broadcast_to(result[:,:1,:50],result[:,:,:50].shape)):
        raise ValueError('Observation-input first50 must be action-invariant')
    return result
