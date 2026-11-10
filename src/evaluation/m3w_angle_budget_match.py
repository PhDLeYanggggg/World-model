"""Match frozen angle selectors by realized replacement count."""

import numpy as np


def match(keep, left, right, left_score, right_score):
    """Rank only existing proposals; keep the original query mask and menu."""
    if (keep.dtype != bool or keep.ndim != 1
            or any(v.shape != keep.shape or not np.isin(v, (0, 1, 2)).all() for v in (left, right))
            or any(v.shape != keep.shape or not np.isfinite(v).all() for v in (left_score, right_score))):
        raise ValueError('Invalid aligned choices or scores')
    pools = [np.flatnonzero(keep & (v != 0)) for v in (left, right)]
    quota = min(map(len, pools))
    output = []
    for pool, choices, score in zip(pools, (left, right), (left_score, right_score), strict=True):
        order = np.argsort(-score[pool], kind='stable')
        chosen = pool[order[:quota]]
        result = np.zeros_like(choices)
        result[chosen] = choices[chosen]
        output.append(result)
    return quota, output, pools


def qualified_expectation(qualified, choice, pool, quota):
    """Expected count for a uniform subset of fixed proposals, no refit."""
    if not 0 <= quota <= len(pool):
        raise ValueError('Impossible matched quota')
    return float(quota * qualified[pool, choice[pool]].mean()) if quota else 0.
