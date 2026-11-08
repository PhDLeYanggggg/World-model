"""Remove the shared part of a fixed intervention, with a norm-matched control."""

import numpy as np


def shared_correction(candidate_velocity, selected, offsets):
    """Return adjusted delta, same-norm direction control, and coherence bin.

    The candidate is OLS4 minus CV, before any intervention mask. The shared
    proxy is the mean of other complete-history agents, not camera pose.
    """
    v = np.asarray(candidate_velocity, dtype=float)
    s, offsets = np.asarray(selected), np.asarray(offsets)
    if (v.ndim != 2 or v.shape[1] != 2 or s.shape != (len(v),) or s.dtype != bool
            or offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != len(v) or (np.diff(offsets) <= 0).any() or not np.isfinite(v).all()):
        raise ValueError("Require aligned finite query-grouped past corrections")
    n = np.diff(offsets)
    total = np.add.reduceat(v, offsets[:-1], axis=0)
    mean = total/n[:, None]
    power = np.add.reduceat(np.square(v).sum(1), offsets[:-1])/n
    coherent = np.divide(np.square(mean).sum(1), power, out=np.zeros_like(power), where=power > 0)
    # -1 distinguishes unsupported singletons from weak measured coherence.
    strata = np.where(n > 1, np.digitize(coherent, [.25, .75]), -1)
    denom = np.repeat(np.maximum(n-1, 1), n)[:, None]
    other = (np.repeat(total, n, axis=0)-v)/denom
    other[np.repeat(n == 1, n)] = 0.
    new = v-other
    raw_norm = np.linalg.norm(v, axis=1)
    norm = np.linalg.norm(new, axis=1)
    cap = np.minimum(raw_norm, norm)
    new *= np.divide(cap, norm, out=np.zeros_like(cap), where=norm > 0)[:, None]
    matched = v*np.divide(cap, raw_norm, out=np.zeros_like(cap), where=raw_norm > 0)[:, None]
    return .5*new*s[:, None], .5*matched*s[:, None], np.repeat(strata, n)
