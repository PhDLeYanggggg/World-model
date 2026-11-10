"""Join a frozen Swiss roster to a caller-provided stream; no source I/O."""

from bisect import bisect_right
import csv

import numpy as np

from src.data_unification.m3w_swiss_roundabout_prefix import HISTORY, project
from src.data_unification.m3w_swiss_roundabout_support import clock_token


def join(stream, prefix, segments, stride):
    ids, off, h, origin = (prefix[k] for k in ('agent_id', 'offsets', 'history', 'origin'))
    groups, current = (prefix[k] for k in ('clock_segment', 'current_tick'))
    n = len(off)-1
    if (type(stride) is not int or stride < 1 or n < 1 or not segments
            or any(not np.issubdtype(a.dtype, np.integer) or a.ndim != 1 for a in (ids, off, groups, current))
            or off[0] != 0 or off[-1] != len(ids) or np.any(np.diff(off) < 2)
            or h.shape != (len(ids), HISTORY, 2) or origin.shape != (n, 2)
            or not np.isfinite(h).all() or not np.isfinite(origin).all() or np.any(ids < 0)
            or groups.shape != (n,) or current.shape != (n,)
            or np.any(groups < 0) or np.any(groups >= len(segments))):
        raise ValueError('Invalid frozen history roster')
    starts = [s['first_clock_centisecond'] for s in segments]
    stops = [s['last_clock_centisecond'] for s in segments]
    if (any(a > b for a, b in zip(starts, stops, strict=True))
            or any(a >= b for a, b in zip(stops[:-1], starts[1:], strict=True))):
        raise ValueError('Overlapping or invalid source clock segments')
    want, seen_current = {}, set()
    for q, (g, t, lo, hi) in enumerate(zip(groups, current, off[:-1], off[1:], strict=True)):
        if (t < (HISTORY-1)*stride or t+12*stride >= segments[g]['ticks']
                or len(set(ids[lo:hi])) != hi-lo
                or q and (g < groups[q-1] or (g == groups[q-1] and t-current[q-1] < 20*stride))):
            raise ValueError('Query overlaps, crosses a clock gap, or repeats an actor')
        for row in range(lo, hi):
            seen_current.add((g, ids[row], t))
            for step in range(12):
                want[g, ids[row], t+(step+1)*stride] = (row, step)
    y = np.zeros((len(ids), 12, 2))
    valid = np.zeros(y.shape[:2], dtype=bool)
    keys, coords, seen_future = [], [], set()
    reader = csv.DictReader(stream)
    if reader.fieldnames != ['track_id', 'type', 'lon', 'lat', 'time']:
        raise ValueError('Unexpected Swiss release columns')
    for r in reader:
        time = clock_token(r['time'])
        group = bisect_right(starts, time)-1
        if group < 0 or time > stops[group]:
            continue
        tick = ((time-starts[group])*3+5)//10
        key = (group, int(r['track_id']), tick)
        if r['lat'] and r['lon']:
            seen_current.discard(key)
        at = want.get(key)
        if at is None:
            continue
        if key in seen_future:
            raise ValueError('Duplicate target tick')
        seen_future.add(key)
        if r['lat'] and r['lon']:
            coords.append((float(r['lat']), float(r['lon'])))
            keys.append(at)
            valid[at] = True
    if seen_current:
        raise ValueError('A frozen actor is missing at the last observed tick')
    if keys:
        rows, steps = np.asarray(keys).T
        anchor = np.repeat(origin, np.diff(off), axis=0)+h[:, -1]
        y[rows, steps] = project(coords)-anchor[rows]
    return y, valid
