"""Outcome-aware zero-harm capacity with one mandatory easy edit."""

import numpy as np


def best_bridge(best, easy, cap=None):
    """Pick a forced easy query and the largest remaining positive gains."""
    positive = np.flatnonzero(np.isfinite(best) & (best > 0.))
    if not len(positive) or cap == 0:
        return 0., None, np.empty(0, dtype=int)
    order = positive[np.argsort(-best[positive], kind="stable")]
    width = len(order) if cap is None else min(cap, len(order))
    gain, pivot, rest = 0., None, np.empty(0, dtype=int)
    for i in np.flatnonzero(np.isfinite(easy) & (easy > 0.)):
        other = order[order != i][:width-1]
        value = float(easy[i]+best[other].sum())
        if value > gain:
            gain, pivot, rest = value, int(i), other
    return gain, pivot, rest
