"""Matched causal cost heads with or without all-current observed-motion context."""
import numpy as np
import torch

from src.data_unification.m3w_moving_forecast_data import SHARED_NAMES
from src.world_model import m3w_bound_aware_cost as base

ARMS = ('local_history', 'shared_motion')
COMMON_DIMENSION = 380
CONTEXT_DIMENSION = len(SHARED_NAMES)
DIMENSION = COMMON_DIMENSION + CONTEXT_DIMENSION


def features(*, geometry, shared_context, rollouts, floor_index, recordings,
             frames, ids, sites):
    """Accept a past-only packet, never outcomes or future-validity masks."""
    h = np.asarray(shared_context)
    if h.shape != (len(geometry), CONTEXT_DIMENSION) or h.dtype != np.float32 or not np.isfinite(h).all():
        raise ValueError('Aligned finite float32 observed-motion48 required')
    causal = base.base.features(geometry=geometry,
        detector_history=np.zeros((len(geometry), 64), np.float32), rollouts=rollouts,
        floor_index=floor_index, recordings=recordings, frames=frames, ids=ids, sites=sites)
    # This source has no admitted detector/image embeddings. Omit their zero slots.
    x = np.concatenate((causal['x'][..., :COMMON_DIMENSION],
        np.broadcast_to(h[:, None], (len(h), 7, CONTEXT_DIMENSION))), axis=-1)
    return dict(causal, x=x)


def for_arm(data, arm):
    if arm not in ARMS or data['z'].shape[-1] != DIMENSION:
        raise ValueError('Registered matched428-column feature treatment required')
    if arm == 'shared_motion':
        return data
    z = data['z'].clone()
    z[..., COMMON_DIMENSION:] = 0
    return dict(data, z=z)


def fit(data, *, arm, identity, **kwargs):
    return base.fit(for_arm(data, arm), arm='bound_aware',
        identity=dict(identity, feature_arm=arm), **kwargs)


def predict(state, causal):
    if set(causal) != base.base.CAUSAL_KEYS or causal['x'].shape[-1] != DIMENSION:
        raise ValueError('Inference accepts the exact causal schema, not future labels')
    arm = state['identity']['feature_arm']
    if arm not in ARMS:
        raise ValueError('Unknown observed-motion feature arm')
    pr = state['preprocess']
    model = base.base.CostHead(DIMENSION, state['settings']['width'],
        state['seed'], pr['bias'], 'context')
    model.load_state_dict(state['model'])
    model.eval()
    result = np.empty((len(causal['x']), 7, 5), np.float64)
    pending, size = [], 0

    def flush():
        ix = np.concatenate(pending)
        segments = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        if arm == 'local_history':
            z[..., COMMON_DIMENSION:] = 0
        scale = causal['scales'][ix]
        envelope = (causal['envelope'][ix]/scale[:, None]).astype(np.float32)
        with torch.no_grad():
            prediction = model(torch.from_numpy(z), torch.from_numpy(envelope),
                torch.from_numpy(segments), len(pending)).numpy()
        result[ix] = prediction.astype(float)*scale[:, None, None]

    for group in causal['groups']:
        if pending and size+len(group) > 1024:
            flush()
            pending, size = [], 0
        pending.append(group)
        size += len(group)
    if pending:
        flush()
    if not np.isfinite(result).all() or (result < 0).any():
        raise FloatingPointError('Finite nonnegative cost moments required')
    return result
