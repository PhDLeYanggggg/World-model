"""Distill feasible recording-level deletion sets into causal query scores."""
import numpy as np

from src.world_model import m3w_record_pruning as base

THRESHOLDS = (0., .01, .025, .05, .1, .2, .35)


def moments(x, membership, deletion_gain, scale, weight):
    w = weight / weight.sum()
    y = np.column_stack((membership, deletion_gain/scale))
    return x.T @ (x*w[:, None]), x.T @ (y*w[:, None])


def scores(x, coef):
    if coef is None:
        return np.full((2, len(x)), -1.)
    imitation, paired_gain = (x @ coef).T
    s = imitation.clip(0., 1.)
    return np.stack((s, np.where(paired_gain >= 0., s, -1.)))


def calibrate(records, arm):
    """Use the same donor eligibility as the original pruner."""
    best = None
    counts = dict(recover_and_preserve_risk=0, also_nonnegative_gain=0)
    for t in THRESHOLDS:
        recovered = lost = extra = count = 0
        gain = 0.
        for group, v, score, full, w in records:
            drop = base.choose(group, score[arm], t)
            old, bad = base.observe(v, np.zeros(len(drop), bool), full)
            new, worse = base.observe(v, drop, full)
            recovered += int(new and not old)
            lost += int(old and not new)
            extra += int(worse and not bad)
            count += int(drop.sum())
            gain += float(w[drop] @ v['remove_lower'][drop])
        if recovered and not lost and not extra:
            counts['recover_and_preserve_risk'] += 1
            if gain >= 0:
                counts['also_nonnegative_gain'] += 1
                option = recovered, gain, -count, t
                best = option if best is None else max(best, option)
    return (None if best is None else best[-1]), counts
