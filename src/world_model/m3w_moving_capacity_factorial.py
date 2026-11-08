"""Causal forecast expansion and exact zero-harm per-query capacity."""

import numpy as np

NAMES = ("ols3", "ols5", "ols6", "theil_sen8", "cv_ols4_midpoint", "ols4_ols8_midpoint", "bounded_accel6")


def slope(h):
    t = np.arange(h.shape[1], dtype=float)
    t -= t.mean()
    return np.sum((h-h[:, -1:])*t[None, :, None], axis=1)/(t@t)


def extra_bank(history, cv, old, ols4, ols8):
    t = np.arange(1., cv.shape[1]+1.)
    i, j = np.triu_indices(history.shape[1], 1)
    robust = np.median((history[:, j]-history[:, i])/(j-i)[None, :, None], axis=1)
    linear = lambda v: t[None, :, None]*v[:, None]
    early, late = slope(history[:, -6:-3]), slope(history[:, -3:])
    a = (late-early)/3.
    limit = .20*np.linalg.norm(late, axis=1)
    a *= np.minimum(1., limit/np.linalg.norm(a, axis=1).clip(1e-12))[:, None]
    # Hold the acquired velocity after four steps rather than extrapolating acceleration indefinitely.
    drift = np.cumsum(np.minimum(t, 4.))
    paths = [*(linear(slope(history[:, -k:])) for k in (3, 5, 6)), linear(robust),
             .5*(cv+ols4), .5*(ols4+ols8), linear(late)+drift[None, :, None]*a[:, None]]
    limit = np.linalg.norm(old-cv[:, None], axis=-1).max((1, 2))
    added = []
    for p in paths:
        delta = p-cv
        size = np.linalg.norm(delta, axis=-1).max(1)
        added.append(cv+delta*np.minimum(1., limit/size.clip(1e-12))[:, None, None])
    return np.stack(added, axis=1)


def plan(gain, easy, groups, cap):
    """Exactly maximize positive weighted gain with a distinct easy anchor."""
    selected, swap = [], None
    for q in groups:
        order = q[gain[q] > 0.]
        order = order[np.argsort(-gain[order], kind="stable")]
        keep = order if cap is None else order[:cap]
        if not len(keep):
            continue
        selected.extend(keep)
        if cap is not None and len(order) > cap and not easy[keep].any():
            reserve = order[easy[order]]
            if len(reserve):
                pair = (gain[keep[-1]]-gain[reserve[0]], int(keep[-1]), int(reserve[0]))
                swap = pair if swap is None else min(swap, pair)
    ids = np.asarray(selected, dtype=int)
    if easy[ids].any():
        return ids
    if swap is None:
        return np.empty(0, dtype=int)
    _, old, new = swap
    ids[ids == old] = new
    return ids
