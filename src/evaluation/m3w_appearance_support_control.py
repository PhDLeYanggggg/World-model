"""Training-only marginal feature support; not a joint-distribution certificate."""
from __future__ import annotations

import numpy as np


def fit_box(training_features):
    x = np.asarray(training_features)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Nonempty finite training feature matrix required')
    return x.min(axis=0), x.max(axis=0)


def project_box(features, lower, upper, columns):
    x = np.asarray(features)
    lower, upper = np.asarray(lower), np.asarray(upper)
    columns = np.asarray(columns)
    if (x.ndim != 2 or lower.shape != (x.shape[1],) or upper.shape != lower.shape
            or not np.isfinite(x).all() or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)
            or columns.ndim != 1 or columns.dtype.kind not in 'iu'
            or np.any(columns < 0) or np.any(columns >= x.shape[1])
            or len(set(columns.tolist())) != len(columns)):
        raise ValueError('Invalid feature box or column indices')
    outside = ((x[:, columns] < lower[columns]) | (x[:, columns] > upper[columns])).any(1)
    result = x.copy()
    result[:, columns] = np.clip(x[:, columns], lower[columns], upper[columns])
    return result, outside
