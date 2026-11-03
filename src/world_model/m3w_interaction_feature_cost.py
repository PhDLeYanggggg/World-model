"""Capacity-matched frozen interaction features versus a zero-feature control."""
import numpy as np
import torch

from src.world_model import m3w_bound_aware_cost as base

ARMS = ('zero64', 'interaction64')


def attach(causal, hidden):
    if set(causal) != base.base.CAUSAL_KEYS:
        raise ValueError('Only causal inputs can receive frozen interaction context')
    h = np.asarray(hidden)
    if h.shape != (len(causal['x']), 64) or h.dtype != np.float32 or not np.isfinite(h).all():
        raise ValueError('Exactly aligned finite float32 ego interaction64 required')
    x = np.concatenate((causal['x'], np.broadcast_to(h[:, None], (*causal['x'].shape[:2], 64))), -1)
    return dict(causal, x=x)


def for_arm(data, arm):
    if arm not in ARMS:
        raise ValueError('Registered interaction64-versus-zero64 comparison required')
    if arm == 'interaction64':
        return data
    z = data['z'].clone()
    z[..., -64:] = 0
    return dict(data, z=z)


def fit(data, *, arm, identity, **kwargs):
    return base.fit(for_arm(data, arm), arm='bound_aware',
                    identity=dict(identity, feature_arm=arm), **kwargs)


def predict(state, causal):
    if set(causal) != base.base.CAUSAL_KEYS:
        raise ValueError('Inference refuses future targets and label support')
    arm = state['identity']['feature_arm']
    if arm not in ARMS:
        raise ValueError('Unknown frozen feature arm')
    pr = state['preprocess']
    model = base.base.CostHead(causal['x'].shape[-1], state['settings']['width'],
                               state['seed'], pr['bias'], 'context')
    model.load_state_dict(state['model'])
    model.eval()
    result = np.empty((len(causal['x']), 7, 5), np.float64)
    pending, size = [], 0
    def flush():
        ix = np.concatenate(pending)
        segments = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        if arm == 'zero64':
            z[..., -64:] = 0
        scale = causal['scales'][ix]
        envelope = (causal['envelope'][ix]/scale[:, None]).astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(z), torch.from_numpy(envelope), torch.from_numpy(segments), len(pending)).numpy()
        result[ix] = p.astype(float)*scale[:, None, None]
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
