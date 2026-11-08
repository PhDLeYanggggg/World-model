"""Optimistic selected-risk bounds for a fixed per-row candidate bank."""

import numpy as np


def lower_bound(error, displacement, baseline_error, known, eligible, easy_cut):
    """Optimistic because denominators include every eligible known row."""
    e, delta, b, known, eligible = map(np.asarray, (error, displacement, baseline_error, known, eligible))
    if (e.ndim != 2 or e.shape != delta.shape or e.shape[1] < 1
            or b.shape != (len(e),) or known.shape != b.shape or eligible.shape != b.shape
            or known.dtype != bool or eligible.dtype != bool or (delta < 0).any()
            or not np.isfinite(delta).all() or not np.isfinite(e[known]).all()
            or not np.isfinite(b[known]).all()):
        raise ValueError("Aligned finite known errors and candidate displacements required")
    easy = known & (b > 0.) & (b <= easy_cut)
    known_take = eligible & known
    easy_take = eligible & easy
    unknown_take = eligible & ~known
    cost = np.where(known[:, None], e, delta)
    choice = cost.argmin(1)
    val = cost[np.arange(len(e)), choice]
    harm = np.maximum(val[known_take]-b[known_take], 0.).sum()
    easy_harm = np.maximum(val[easy_take]-b[easy_take], 0.).sum()
    missing = val[unknown_take].sum()
    reference = b[known_take].sum()
    easy_reference = b[easy_take].sum()
    ratio = lambda n, d: float(n/d) if d > 0. else None
    return choice, dict(
        all_lower=ratio(harm+missing, reference), easy_lower=ratio(easy_harm+missing, easy_reference),
        signed_easy_lower=ratio((val[easy_take]-b[easy_take]).sum(), b[easy].sum()))
