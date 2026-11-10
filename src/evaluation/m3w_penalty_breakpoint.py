"""Determine when a frozen nonnegative penalty changes a finite-menu winner."""

import numpy as np


def breakpoint(utility, penalty, weight):
    u, p = np.asarray(utility, dtype=float), np.asarray(penalty, dtype=float)
    if (u.ndim != 1 or not len(u) or p.shape != u.shape or not np.isfinite(u).all()
            or not np.isfinite(p).all() or np.any(p < 0.) or not np.isfinite(weight) or weight < 0.):
        raise ValueError('Expected a finite menu and a nonnegative penalty weight')
    base = int(np.argmax(u))
    winner = int(np.argmax(u-weight*p))
    relief = p[base]-p
    can_improve = relief > 0.
    threshold = float(np.min((u[base]-u[can_improve])/relief[can_improve])) if can_improve.any() else None
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError('Nonfinite penalty crossover')
    return dict(base=base, winner=winner, threshold=threshold, penalty_at_base=float(p[base]),
                alternatives_with_less_penalty=int(can_improve.sum()))
