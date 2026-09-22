"""Fixed past-forecast-only temporal intervention and displacement-matched control."""
import numpy as np
from src.evaluation.m3w_conditional_cost_audit import strict_bits

ARMS = ('ramp', 'uniform')
POLICIES = ('ramp_strict', 'uniform_strict', 'uniform_matched', 'ramp_at_uniform',
            'uniform_at_ramp', 'ramp_uncontrolled', 'uniform_uncontrolled')


def candidates(baseline, neural):
    b, p = np.asarray(baseline, float), np.asarray(neural, float)
    if b.ndim != 3 or b.shape[1:] != (12, 2) or p.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('Finite aligned 12-step forecasts required; no targets accepted')
    delta = p-b
    distance = np.linalg.norm(delta, axis=-1)
    weights = np.arange(12, dtype=float)/11
    den = distance.sum(1)
    alpha = np.divide((distance*weights).sum(1), den, out=np.zeros(len(b)), where=den>0)
    ramp = b + weights[None, :, None]*delta
    uniform = b + alpha[:, None, None]*delta
    return dict(ramp=ramp, uniform=uniform), alpha


def selections(ramp, uniform, past, distance, ids):
    if (distance.shape != (len(ids),) or len(np.unique(ids)) != len(ids)
            or ramp.shape != (len(ids), 2) or uniform.shape != ramp.shape):
        raise ValueError('Aligned scores, common causal disagreement and unique ids required')
    r, u = strict_bits(ramp, past, distance), strict_bits(uniform, past, distance)
    support = (distance>0) & np.any(past[:, -1] != past[:, -2], axis=1)
    pool = np.flatnonzero(support)
    gain = uniform[:, 0]-uniform[:, 1]
    matched = np.zeros(len(ids), bool)
    matched[pool[np.lexsort((ids[pool], -gain[pool]))[:int(r.sum())]]] = True
    return dict(ramp_strict=r, uniform_strict=u, uniform_matched=matched,
                ramp_at_uniform=u.copy(), uniform_at_ramp=r.copy(),
                ramp_uncontrolled=support.copy(), uniform_uncontrolled=support.copy())


def policy_arm(name):
    if name not in POLICIES:
        raise ValueError('Unregistered policy')
    return 'ramp' if name.startswith('ramp') else 'uniform'


def second_difference(past, future, scale):
    """Discrete native-pixel smoothness proxy, not acceleration in meters/seconds."""
    h, p, s = np.asarray(past, float), np.asarray(future, float), np.asarray(scale, float)
    if (h.shape != (len(p), 8, 2) or p.shape[1:] != (12, 2) or s.shape != (len(p),)
            or not np.isfinite(h).all() or not np.isfinite(p).all()
            or not np.isfinite(s).all() or np.any(s<=0)):
        raise ValueError('Finite paired past/future rollouts and positive causal scale required')
    return np.linalg.norm(np.diff(np.concatenate((h[:, -2:], p), 1), n=2, axis=1), axis=-1).mean(1)*s
