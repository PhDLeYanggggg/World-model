"""Exact relative-motion bounds for a small set of trajectory edits."""

import numpy as np


def compare(prediction, reference, target, valid):
    if (prediction.shape != reference.shape or target.shape != reference.shape or reference.ndim != 3
            or reference.shape[-1] != 2 or valid.shape != reference.shape[:2] or valid.dtype != bool
            or not np.isfinite(prediction).all() or not np.isfinite(reference).all()
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Expected aligned forecasts and masked positions')
    n, steps, _ = reference.shape
    total = n*(n-1)*steps//2
    count = valid.sum(0)
    observed = int((count*(count-1)//2).sum())
    changed = np.any(prediction != reference, axis=(1, 2))
    if n < 2 or not steps or not changed.any():
        return dict(lower=0., upper=0., available=0. if observed else None, pair_steps=total,
                    observed_pair_steps=observed, affected_pairs=0)
    # For two competing two-edit slates, at most four actors can differ.
    left, right = [], []
    for a in np.flatnonzero(changed):
        b = np.flatnonzero((np.arange(n) != a) & (~changed | (np.arange(n) > a)))
        left.extend([a]*len(b))
        right.extend(b.tolist())
    i, j = np.asarray(left), np.asarray(right)
    known = valid[i] & valid[j]
    x, r = prediction[i]-prediction[j], reference[i]-reference[j]
    truth = np.where(known[..., None], target[i]-target[j], 0.)
    gain = np.linalg.norm(r-truth, axis=-1)-np.linalg.norm(x-truth, axis=-1)
    score = float(np.where(known, gain, 0.).sum())
    missing = float(np.where(known, 0., np.linalg.norm(x-r, axis=-1)).sum())
    return dict(lower=(score-missing)/total, upper=(score+missing)/total,
                available=score/observed if observed else None, pair_steps=total,
                observed_pair_steps=observed, affected_pairs=len(i))
