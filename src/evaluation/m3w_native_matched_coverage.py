"""Outcome-blind, exact-count controls for frozen native cost scores."""
from __future__ import annotations

import hashlib
import numpy as np


def top_count(score, eligible, ids, count):
    value, mask, ids = np.asarray(score), np.asarray(eligible), np.asarray(ids)
    if (value.ndim != 1 or mask.shape != value.shape or mask.dtype != bool
            or ids.shape != value.shape or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or not np.isfinite(value).all()
            or isinstance(count, bool) or int(count) != count or not 0 <= count <= mask.sum()):
        raise ValueError('Finite scores, unique query IDs and supported exact count required')
    rows = np.flatnonzero(mask)
    order = np.lexsort((ids[rows], -value[rows]))
    selected = np.zeros(len(value), bool)
    selected[rows[order[:int(count)]]] = True
    return selected


def ratio_score(cost):
    p = np.asarray(cost, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Finite nonnegative paired cost scores required')
    total = p.sum(1)
    return np.divide(p[:, 0]-p[:, 1], total, out=np.zeros(len(p)), where=total > 0)


def hash_priority(ids, site, seed):
    # A fixed shuffle of sorted IDs is invariant to input order and label support.
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique query IDs required')
    salt = int.from_bytes(hashlib.sha256(f'native-coverage-v1:{site}:{seed}'.encode()).digest()[:8], 'little')
    ranks = np.random.default_rng(salt).permutation(len(ids))
    score = np.empty(len(ids), np.float64)
    score[np.argsort(ids)] = ranks
    return score


def matched_policies(scores, same, ids, *, site, seed, budget):
    """No outcomes or support masks enter budget, ordering or tie breaking."""
    if budget not in ('positive_gain', 'harm_fraction_0p1'):
        raise ValueError('Only the two frozen reference budgets are permitted')
    same = np.asarray(same)
    a, m, r = (np.asarray(scores[k]) for k in ('underharm4', 'mse', 'ridge_raw'))
    if (same.dtype != bool or same.shape != (len(a),) or m.shape != a.shape or r.shape != a.shape
            or not np.isfinite(r).all()):
        raise ValueError('Aligned causal scores and rollout identity mask required')
    ar, mr = ratio_score(a), ratio_score(m)
    anchor = (a[:, 0] > a[:, 1]) & ~same
    if budget == 'harm_fraction_0p1':
        anchor &= a[:, 1] <= .1*a[:, 0]
    count = int(anchor.sum())
    # Exact reproduction checks the bridge between threshold and ranking controls.
    np.testing.assert_array_equal(top_count(ar, ~same, ids, count), anchor)
    rankings = dict(mse_ratio=mr, mse_net_gain=m[:, 0]-m[:, 1],
        asym_net_gain=a[:, 0]-a[:, 1], ridge_raw_net_gain=r[:, 0]-r[:, 1],
        hash_control=hash_priority(ids, site, seed))
    return dict(asym_rule=anchor, **{k:top_count(v, ~same, ids, count) for k,v in rankings.items()})


def paired_scene_contrast(scene_gains_a, scene_gains_b, *, resamples=3000, seed=38113):
    a, b = np.asarray(scene_gains_a, float), np.asarray(scene_gains_b, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired finite physical-scene gains required')
    difference = a-b
    samples = np.random.default_rng(seed).choice(difference, size=(resamples, len(a))).mean(1)
    return dict(mean_gain_difference_pp=float(difference.mean()),
        scene_differences_pp=difference.tolist(), ci95_pp=np.quantile(samples, [.025, .975]).tolist(),
        resamples=resamples, seed=seed, unit='physical_scene',
        interpretation='conditional_development_contrast_not_independent_confirmation')
