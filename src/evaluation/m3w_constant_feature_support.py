"""A fixed diagnostic for extrapolation in exactly constant training dimensions."""
import numpy as np


def clamp_constant_support(train, queries):
    train, queries = np.asarray(train), np.asarray(queries)
    if train.ndim != 2 or queries.ndim != 2 or train.shape[1] != queries.shape[1] or not len(train):
        raise ValueError('Nonempty compatible feature matrices required')
    if not np.isfinite(train).all() or not np.isfinite(queries).all():
        raise ValueError('Finite source features required')
    constant = np.ptp(train, axis=0) == 0
    repaired = queries.copy()
    repaired[:, constant] = train[0, constant]
    return repaired, constant
