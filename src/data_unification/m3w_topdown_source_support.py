"""Structural 8/12 support from source-local track IDs and sample indices."""

import numpy as np


def grid_indices(timestamps, tick=1001. / 30000., tolerance=6e-7):
    t = np.asarray(timestamps, dtype=np.float64)
    if not np.isfinite(t).all() or (t < 0.).any() or tick <= 0:
        raise ValueError("Invalid source timestamps")
    index = np.rint(t / tick).astype(np.int64)
    residual = np.abs(t - index*tick)
    if (residual > tolerance).any():
        raise ValueError(f"Off-grid timestamps (max residual {residual.max():.9f})")
    return index, float(residual.max(initial=0.))


def support(tracks, frame_count, stride):
    if type(frame_count) is not int or frame_count < 1 or type(stride) is not int or stride < 1:
        raise ValueError("Positive source length and stride required")
    n = max(0, frame_count - 19*stride)
    past = np.zeros(n, dtype=np.int64)
    complete = np.zeros_like(past)
    values = dict(past_eligible=0, endpoint=0, complete12=0, partial12=0, no_future=0,
                  eligible_tracks=0, complete_tracks=0)
    for times in tracks:
        a = np.asarray(times, dtype=np.int64)
        if not len(a) or (a < 0).any() or (a >= frame_count).any() or (np.diff(a) <= 0).any():
            raise ValueError("Expected sorted unique frame indices per track")
        if not n:
            continue
        first = a[a <= n-1]
        observed = np.zeros(frame_count, dtype=bool)
        observed[a] = True
        keep = first[np.logical_and.reduce([observed[first+k*stride] for k in range(1, 8)])]
        if not len(keep):
            continue
        future = np.stack([observed[keep+k*stride] for k in range(8, 20)])
        full, any_ = future.all(0), future.any(0)
        past[keep] += 1
        complete[keep[full]] += 1
        values["past_eligible"] += int(len(keep))
        values["endpoint"] += int(future[-1].sum())
        values["complete12"] += int(full.sum())
        values["partial12"] += int((any_ & ~full).sum())
        values["no_future"] += int((~any_).sum())
        values["eligible_tracks"] += 1
        values["complete_tracks"] += int(full.any())
    values.update(queries=int((past > 0).sum()), multiagent_queries=int((past >= 2).sum()),
                  multiagent_complete_queries=int((complete >= 2).sum()))
    free = nonoverlapping = 0
    for i in np.flatnonzero(complete >= 2):
        if i >= free:
            free = i + 19*stride + 1
            nonoverlapping += 1
    values["nonoverlapping_multiagent_complete"] = nonoverlapping
    return values
