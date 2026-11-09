"""Add a forecast-independent easy-support term to a frozen pair score."""

import numpy as np


def rank_with_support(merit, first, second, logits):
    support = np.maximum(logits[first], logits[second])
    if support.shape != merit.shape or not (np.isfinite(support).all() and np.isfinite(merit).all()):
        raise ValueError('Expected aligned finite pair scores and actor support logits')
    # Use the same negative-log penalty weight as the existing risk terms.
    return merit - .25*np.logaddexp(0., -support)


def shuffle_by_query(logits, counts, seed):
    if sum(counts) != len(logits) or np.any(counts <= 0):
        raise ValueError('Counts must describe the frozen roster')
    rng = np.random.default_rng(seed)
    result = logits.copy()
    lo = 0
    for n in counts:
        result[lo:lo+n] = rng.permutation(logits[lo:lo+n])
        lo += n
    return result
