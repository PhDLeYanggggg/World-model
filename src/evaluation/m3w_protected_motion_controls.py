"""Past-only candidate extraction and outcome-blind protected-motion controls."""
import numpy as np

from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation.m3w_native_matched_coverage import top_count

ACTIONS = tuple(v for v in BASELINES if v != 'constant_velocity_causal_fd') + ('transformer',)


def causal_candidate(geometry, action):
    g = np.asarray(geometry)
    if (g.ndim != 2 or g.shape[1] != 476 or not np.isfinite(g).all()
            or action not in ACTIONS[:-1]):
        raise ValueError('Frozen finite past-only geometry and causal action required')
    k = BASELINES.index(action)
    return g[:, 308+24*k:308+24*(k+1)].reshape(-1, 12, 2).copy()


def protected_decisions(scores, history, distance):
    score, past, d = map(np.asarray, (scores, history, distance))
    if (score.shape != (len(past), 2) or past.shape != (len(score), 8, 2)
            or d.shape != (len(score),) or not np.isfinite(score).all()
            or not np.isfinite(past).all() or not np.isfinite(d).all()
            or (score < 0).any() or (d < 0).any()
            or np.any(score.sum(1) > d+2e-6*(1+d))):
        raise ValueError('Finite bounded costs and causal support required')
    support = (d > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
    net = support & (score[:, 0] > score[:, 1])
    return dict(net_stop=net, strict_stop=net & (score[:, 1] <= .1*score[:, 0]))


def match_strict_count(left_score, right_score, left_bits, right_bits, ids):
    """Match inside each strict eligible pool without observing outcome labels."""
    a, b = np.asarray(left_score), np.asarray(right_score)
    count = min(int(np.asarray(left_bits).sum()), int(np.asarray(right_bits).sum()))
    return (top_count(a[:, 0]-a[:, 1], left_bits, ids, count),
            top_count(b[:, 0]-b[:, 1], right_bits, ids, count))


def supported_costs(candidate_ade, cv_ade, full):
    a, b, complete = map(np.asarray, (candidate_ade, cv_ade, full))
    if (a.ndim != 1 or b.shape != a.shape or complete.shape != a.shape
            or complete.dtype != bool or not np.isfinite(a[complete]).all()
            or not np.isfinite(b[complete]).all() or (a[complete] < 0).any()
            or (b[complete] < 0).any()):
        raise ValueError('Aligned complete-path nonnegative supervision required')
    delta = b-a
    y = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
    y[~complete] = np.nan
    return y, np.where(complete, b, np.nan)
