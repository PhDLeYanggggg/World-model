"""Causal packet reconstruction with no outcome fields in the interface."""
import numpy as np

from src.world_model import m3w_shared_motion_cost as shared
from src.world_model import m3w_amplitude_cost as amplitude
from src.world_model import m3w_rollout_history_risk as relation
from src.world_model import m3w_candidate_neighbor_basis as neighbors
from src.world_model.m3w_european_source_forecast import baseline_numpy


def packet(*, geometry, shared_context, history, recordings, frames, ids, sites,
           relative_forecast, floor_index):
    h = np.asarray(history)
    origin = h[:,-1]
    physical = np.stack([baseline_numpy(h,k)-origin[:,None] for k in range(6)],1)
    paths = np.concatenate((physical,np.asarray(relative_forecast,dtype=float)[:,None]),1)
    causal = shared.features(geometry=geometry, shared_context=shared_context, rollouts=paths,
        floor_index=floor_index, recordings=recordings, frames=frames, ids=ids, sites=sites)
    causal['allowed'][:,:6] = False
    return amplitude.causal_bank(old_causal=causal,geometry=geometry,shared_context=shared_context,
                                 paths=paths,floor_index=floor_index)


def summary(causal, proposal):
    active = np.flatnonzero(proposal>=0)
    x = causal['x']
    # These are the original raw120 features; no new fit or normalization here.
    parts = [relation.parent.kin.motion_basis(x), relation.basis(x), neighbors.chunked(x)]
    common = np.concatenate(parts,-1)
    result = common[active,proposal[active]].astype(float)
    if result.shape!=(len(active),120) or not np.isfinite(result).all():
        raise ValueError('Finite original120 for the causal proposal cohort required')
    return active,result
