"""Causal linear motion controls and observed-prefix selection on one query."""

import numpy as np


ARMS = ("cv", "ols4", "median4", "fitted_origin", "pooled_origin", "past_best", "past_guarded")
PREF = (1, 0, 2, 3, 4)
PAST_LIMIT = .02
II, JJ = np.triu_indices(4, k=1)


def base(history):
    """Each result stores a relative origin and a constant slope."""
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1] < 4 or h.shape[2] != 2 or len(h) == 0 or not np.isfinite(h).all():
        raise ValueError("Expected a complete causal 2D history for each requested actor")
    tail = h[:, -4:]-h[:, -1:]
    v = np.einsum("ntd,t->nd", tail, [-1.5, -.5, .5, 1.5])/5.
    offset = tail.mean(1)+1.5*v
    median = np.median((tail[:, JJ]-tail[:, II])/(JJ-II)[None, :, None], axis=1)
    out = np.zeros((len(h), 5, 2, 2))
    out[:, :, 1] = np.stack((-tail[:, -2], v, median, v, v), axis=1)
    out[:, 3, 0] = offset
    out[:, 4, 0] = np.median(offset, axis=0)
    return out


def coefficients(history):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2):
        raise ValueError("Expected the fixed eight-step history")
    full = base(h)
    errors = []
    for k in (4, 5, 6, 7):
        b = base(h[:, :k])
        actual = h[:, k]-h[:, k-1]
        errors.append(np.linalg.norm(b[:, :, 0]+b[:, :, 1]-actual[:, None], axis=-1))
    error = np.stack(errors, axis=2)
    mean = error.mean(2)
    order = np.asarray(PREF)
    raw = order[mean[:, order].argmin(1)]
    cv = error[:, :1]
    safe = (error <= (1.+PAST_LIMIT)*cv+1e-12).all(2) & (mean < mean[:, :1]-1e-12)
    best = np.where(safe[:, order], mean[:, order], np.inf).argmin(1)
    guarded = np.where(safe.any(1), order[best], 0)
    i = np.arange(len(h))
    return np.concatenate((full, full[i, raw, None], full[i, guarded, None]), axis=1)


def paths(coeff, horizon=12):
    if coeff.shape[-2:] != (2, 2) or not np.isfinite(coeff).all():
        raise ValueError("Invalid frozen linear coefficients")
    return coeff[..., 0, None, :]+coeff[..., 1, None, :]*np.arange(1., horizon+1.)[:, None]
