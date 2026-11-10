"""Bound a direct contrast when repeated seeds share the same missing target."""

from itertools import permutations

import numpy as np


def shared_geometry(ref, pred, missing, weight):
    if (ref.shape != pred.shape or ref.shape != (3, *missing.shape, 2)
            or missing.ndim != 2 or missing.dtype != bool or weight.shape != missing.shape[:1]
            or not all(np.isfinite(x).all() for x in (ref, pred, weight)) or np.any(weight <= 0.)):
        raise ValueError('Need three aligned forecast seeds and original query weights')
    a, b = ref[:, missing], pred[:, missing]
    w = np.broadcast_to(weight[:, None], missing.shape)[missing]
    d = np.linalg.norm(a[:, None]-b[None], axis=-1)
    diag = d[np.arange(3), np.arange(3)]
    diagonal = diag.mean(0)
    coupled = np.minimum.reduce([d[np.arange(3), p].mean(0) for p in permutations(range(3))])
    # A shared point at one of the six forecasts is a feasible stepwise witness.
    y = np.concatenate((a, b), axis=0)
    change = (np.linalg.norm(a[:, None]-y[None], axis=-1)-np.linalg.norm(b[:, None]-y[None], axis=-1)).mean(0)
    lower, upper = change.min(0), change.max(0)
    if (np.any(coupled > diagonal+1e-10) or np.any(lower < -coupled-1e-10)
            or np.any(upper > coupled+1e-10)):
        raise ValueError('Coupling or witness exceeds its certified bound')
    return dict(per_seed_diagonal=(diag @ w).tolist(), shared=float(w @ coupled),
                witness_low=float(w @ lower), witness_high=float(w @ upper),
                affected_steps=int((diagonal > 1e-12).sum()), tighter_steps=int((diagonal-coupled > 1e-12).sum()))
