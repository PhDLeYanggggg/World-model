"""Select recipients by additive merit with their ordered path families fixed."""

import numpy as np

from src.world_model.m3w_vehicle_pair_learning import merit


def best_pair(head, legal, families):
    """`head` holds per-recipient four-channel outputs, before pair averaging."""
    if head.ndim != 3 or head.shape[:2] != legal.shape or head.shape[2] != 4 or len(families) != 2:
        raise ValueError('Expected per-actor outputs and two ordered path families')
    f, g = map(int, families)
    if not 0 <= f < legal.shape[1] or not 0 <= g < legal.shape[1]:
        raise ValueError('Invalid ordered path family')
    left, right = np.flatnonzero(legal[:, f]), np.flatnonzero(legal[:, g])
    i, j = np.repeat(left, len(right)), np.tile(right, len(left))
    mask = i < j if f == g else i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        raise ValueError('There must be a legal pair at the reference query')
    scores = merit(.5*(head[i, f]+head[j, g]))
    if not np.isfinite(scores).all():
        raise ValueError('Invalid additive score')
    k = int(scores.argmax())
    return np.array([i[k], j[k]]), len(i)
