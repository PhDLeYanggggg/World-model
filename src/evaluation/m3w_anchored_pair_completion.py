"""Use observed endpoints to share one missing target across different edges."""
import numpy as np


def occurrence_terms(left, right, frames, paths, left_y, right_y, left_valid, right_valid):
    one = left_valid ^ right_valid
    two = ~left_valid & ~right_valid
    sign = np.where(left_valid, -1., 1.)
    observed = np.where(left_valid[..., None], left_y, right_y)
    # If the left endpoint is observed, flip the edge. Then
    # |(prediction + observed_endpoint) - missing| is the original error.
    missing = np.where(left_valid, right[:, None], left[:, None])
    if np.any(missing[one] == np.iinfo(np.int64).max):
        raise ValueError('Invalid agent ID for the single-target key')
    key = np.empty((*one.shape, 3), dtype=np.int64)
    key[..., 0], key[..., 1], key[..., 2] = missing, missing+1, frames
    single = paths[:, one]*sign[one][None, :, None]+observed[one][None]
    other = key.copy()
    other[..., 0] = np.minimum(left, right)[:, None]
    other[..., 1] = np.maximum(left, right)[:, None]
    p = paths[:, two]*np.broadcast_to(np.where(left < right, 1., -1.)[:, None], two.shape)[two][None, :, None]
    return {'one_missing': (key[one], single), 'two_missing': (other[two], p)}, (int(one.sum()), int(two.sum()))
