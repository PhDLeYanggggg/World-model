"""Past-only risk-support scores for unchanged two-actor proposals."""

import numpy as np


ARMS = ('original', 'supported_merit', 'support_only')


def scores(merit, p, rows):
    if (p.ndim != 1 or rows.shape != (len(merit), 2) or rows.dtype.kind not in 'iu'
            or np.any(rows < 0) or np.any(rows >= len(p)) or np.any(rows[:, 0] == rows[:, 1])
            or not np.isfinite(merit).all() or not np.all((p >= 0.) & (p <= 1.))):
        raise ValueError('Require finite frozen scores and two distinct causal actors per proposal')
    a, b = p[rows].T.astype(np.float64)
    support = a+b-a*b
    return np.stack((merit, merit+np.log(np.maximum(support, 1e-12)), support))


def quota(scores):
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid proposal score roster')
    used = np.zeros(len(scores), dtype=bool)
    used[np.argsort(-scores, kind='stable')[:max(1, len(scores)//4)]] = True
    return used


def prediction(cv, rows, paths, use):
    if use.shape != (len(rows),) or use.dtype != bool or len(np.unique(rows)) != rows.size:
        raise ValueError('Expected one distinct two-actor proposal per recording')
    out = cv.copy()
    out[rows[use]] = paths[use]
    return out
