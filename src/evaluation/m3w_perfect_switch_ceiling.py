"""Outcome-only bounds under a fixed per-query number of changed forecasts."""

import numpy as np


def capped_masses(gain, offsets, caps):
    """Maximize additive nonnegative gain at each query's fixed cardinality."""
    gain = np.asarray(gain, dtype=float)
    offsets, caps = np.asarray(offsets), np.asarray(caps)
    lengths = np.diff(offsets)
    if (gain.ndim != 1 or not np.isfinite(gain).all() or np.any(gain < 0.)
            or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(gain)
            or np.any(lengths <= 0) or caps.ndim != 2 or caps.shape[0] != len(lengths)
            or caps.dtype.kind not in "iu" or np.any(caps < 0)):
        raise ValueError("Finite gains, nonempty aligned queries and integral caps required")
    result = np.zeros(caps.shape[1], dtype=float)
    for lo, hi, k in zip(offsets[:-1], offsets[1:], caps, strict=True):
        ordered = np.sort(gain[lo:hi])[::-1]
        total = np.r_[0., np.cumsum(ordered)]
        result += total[np.minimum(k, hi-lo)]/(hi-lo)
    return result


def effective_caps(base, offsets, slots, paths):
    """Count changed paths rather than nominations or identity replacements."""
    slots, paths, offsets = map(np.asarray, (slots, paths, offsets))
    if (slots.ndim != 2 or slots.shape[0] != len(offsets)-1 or slots.dtype.kind not in "iu"
            or paths.shape != (*slots.shape, *base.shape[1:]) or np.any(slots < -1)):
        raise ValueError("Aligned sparse query slots required")
    used = slots >= 0
    if (np.any(used & ((slots < offsets[:-1, None]) | (slots >= offsets[1:, None])))
            or len(np.unique(slots[used])) != int(used.sum())):
        raise ValueError("Slots must identify distinct rows in their own query")
    changed = np.zeros_like(used)
    changed[used] = np.any(paths[used] != base[slots[used]], axis=(-2, -1))
    return changed.sum(axis=1)
