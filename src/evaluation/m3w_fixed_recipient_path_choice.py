"""Rank the same legal path pairs on predeclared two-recipient sets."""

import numpy as np


def select(row, i, j, scores, recipients):
    if (row.ndim != 1 or i.shape != j.shape or i.ndim != 1
            or not np.issubdtype(i.dtype, np.integer) or not np.issubdtype(j.dtype, np.integer)
            or (i < 0).any() or (j < 0).any() or (i >= len(row)).any() or (j >= len(row)).any()
            or (row[i] == row[j]).any() or len(set(recipients)) != 2):
        raise ValueError('Require legal pairs of distinct recipients')
    target = np.sort(recipients)
    a, b = row[i], row[j]
    take = np.flatnonzero((np.minimum(a, b) == target[0]) & (np.maximum(a, b) == target[1]))
    if not len(take):
        raise ValueError('Frozen recipients have no legal common path menu')
    out = {}
    for k, score in scores.items():
        if score.shape != i.shape or not np.isfinite(score).all():
            raise ValueError('Invalid frozen scores')
        best = take[np.argmax(score[take])]
        out[k] = (np.array([i[best], j[best]]), float(score[best]))
    return out
