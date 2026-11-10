"""Use predicted label missingness only to rank fixed forecast edits."""

import numpy as np


def normalized_debit(distance, missing):
    """Per-proposal expected debit, normalized by the two peak radii."""
    if (distance.ndim != 3 or distance.shape[1:] != (2, 12) or missing.shape != distance.shape
            or not np.isfinite(distance).all() or np.any(distance < 0.)
            or not np.all((0. <= missing) & (missing <= 1.))):
        raise ValueError('Expected two actors, twelve finite steps and valid probabilities')
    scale = 12.*distance.max(axis=2).sum(axis=1)
    if np.any(scale <= 0.):
        raise ValueError('An edited proposal must differ from the fallback')
    return (distance*missing).sum(axis=(1, 2))/scale
