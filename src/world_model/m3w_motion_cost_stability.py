"""Matched two-recipient contests for source sensitivity of cost estimates."""

import numpy as np


def contests(query, radius, legal, full_score):
    """Adjacent past-radius pairs; retain one contest per query by full-fit gain."""
    if not (len(query) == len(radius) == len(legal) == len(full_score)):
        raise ValueError('Misaligned query inputs')
    order = np.lexsort((np.arange(len(query)), radius, query))
    order = order[legal[order]]
    split = np.flatnonzero(query[order][1:] != query[order][:-1]) + 1
    pairs = []
    for ix in np.split(order, split):
        pair = ix[:2 * (len(ix) // 2)].reshape(-1, 2)
        if not len(pair):
            continue
        shared = radius[pair[:, 0]]
        comparable = (shared > 0.) & (radius[pair[:, 1]] <= 2 * shared)
        if comparable.any():
            pair, shared = pair[comparable], shared[comparable]
            pairs.append(pair[np.argmax(full_score[pair].max(1) * shared)])
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def decide(full, refits, pairs):
    """Every arm chooses exactly one member of each fixed pair."""
    if refits.ndim != 2 or refits.shape[1] != len(full) or len(refits) < 2:
        raise ValueError('Require source-exclusion scores for all rows')
    if not np.isfinite(refits).all() or not np.isfinite(full).all():
        raise ValueError('Nonfinite score')
    row = np.arange(len(pairs))
    return {name: pairs[row, value[pairs].argmax(1)] for name, value in
            dict(full_fit=full, omit_mean=refits.mean(0), omit_min=refits.min(0)).items()}


def edited_path(ols, picked, radius, pairs, selected):
    if not np.any(pairs == selected[:, None], axis=1).all() or len(np.unique(selected)) != len(pairs):
        raise ValueError('Require one recipient per fixed contest')
    shared = radius[pairs].min(1)
    result = ols.copy()
    result[selected] += (picked[selected] - ols[selected]) * (shared / radius[selected])[:, None, None]
    peak = np.linalg.norm(result - ols, axis=-1).max(1)
    if np.count_nonzero(peak) != len(pairs):
        raise ValueError('Edit count changed')
    np.testing.assert_allclose(peak[selected], shared, rtol=1e-8, atol=1e-10)
    return result
