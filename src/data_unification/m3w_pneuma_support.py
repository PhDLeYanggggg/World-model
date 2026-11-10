"""Structural pNEUMA coverage; no coordinates after track prefix are decoded."""

from collections import Counter
from dataclasses import dataclass
import csv
import hashlib

import numpy as np


HEADER = ('track_id', 'type', 'traveled_d', 'avg_speed', 'lat', 'lon', 'speed',
          'lon_acc', 'lat_acc', 'time')
TICK = .04
ABSENT = frozenset(('', 'nan', 'na', 'null', 'none'))


@dataclass
class Tracks:
    time: list
    position_present: list
    ids: set
    prefixes: dict
    first_points: list
    types: Counter
    intervals: Counter
    diagnostics: Counter


def scan(source):
    rows = csv.reader(source, delimiter=';')
    head = tuple(c.strip().lower() for c in next(rows))
    if head[:len(HEADER)] != HEADER or any(c for c in head[len(HEADER):]):
        raise ValueError(f'Unexpected vehicle CSV header: {head}')
    tracks = Tracks([], [], set(), {}, [], Counter(), Counter(), Counter())
    for row in rows:
        if not row or not any(c.strip() for c in row):
            continue
        if (len(row)-4) % 6 == 1 and not row[-1].strip():
            row = row[:-1]
        if len(row) < 10 or (len(row)-4) % 6:
            raise ValueError('Partial vehicle track tuple')
        agent = int(row[0])
        if agent in tracks.ids:
            raise ValueError('Duplicate publisher track within a file')
        tracks.ids.add(agent)
        kind = row[1].strip()
        if not kind:
            raise ValueError('Missing vehicle type')
        tracks.types[kind] += 1
        times = np.asarray(row[9::6], dtype=np.float64)
        if not np.isfinite(times).all() or (times < 0.).any() or (np.diff(times) <= 0.).any():
            raise ValueError('Expected strictly increasing nonnegative times')
        micros = np.rint(np.diff(times)*1e6).astype(np.int64)
        u, n = np.unique(micros, return_counts=True)
        tracks.intervals.update(dict(zip(u.tolist(), n.tolist(), strict=True)))
        clock = np.rint(times/TICK).astype(np.int64)
        on_grid = np.abs(times-clock*TICK) <= 1e-6
        tracks.diagnostics.update(rows=1, observations=len(times), off_grid=int((~on_grid).sum()))
        # Presence is checked textually. Coordinates outside the prefix are not decoded.
        present = np.fromiter((x.strip().lower() not in ABSENT and y.strip().lower() not in ABSENT
                               for x, y in zip(row[4::6], row[5::6], strict=True)), dtype=bool, count=len(times))
        tracks.diagnostics.update(position_tokens_absent=int((~present).sum()))
        tracks.time.append(clock[on_grid])
        tracks.position_present.append(present[on_grid])
        n = min(8, len(times))
        first = [(float(row[i]), float(row[i+1])) for i in range(4, 4+6*n, 6)
                 if present[(i-4)//6]]
        if any(not (-90 <= lat <= 90 and -180 <= lon <= 180) for lat, lon in first):
            raise ValueError('Invalid observed prefix coordinate')
        if first:
            tracks.first_points.append(first[0])
        if n == 8 and present[:n].all():
            stamp = tuple(np.rint(times[:n]*1e6).astype(np.int64).tolist())
            pos = tuple((round(lat, 6), round(lon, 6)) for lat, lon in first)
            key = hashlib.sha256(repr((stamp, pos)).encode()).digest()
            tracks.prefixes[key] = tracks.prefixes.get(key, 0)+1
    if not tracks.ids:
        raise ValueError('Empty pilot member')
    return tracks


def coverage(tracks, stride):
    if type(stride) is not int or stride < 1:
        raise ValueError('Expected positive stride')
    clock = [t for t in tracks.time if len(t)]
    if not clock:
        raise ValueError('No verified release clock')
    lo, hi = min(t[0] for t in clock), max(t[-1] for t in clock)
    span = hi-lo+1
    if span > 100_000:
        raise ValueError('Pilot clock span exceeds declared extraction bound')
    width = max(0, span-19*stride)
    roster = np.zeros(width, dtype=np.int64)
    past_rows = []
    for time, valid in zip(tracks.time, tracks.position_present, strict=True):
        use = time[valid]-lo
        if (np.diff(use) <= 0).any():
            raise ValueError('On-grid duplicates')
        has = np.zeros(span, dtype=bool)
        has[use] = True
        start = use[use < width]
        keep = start[np.logical_and.reduce([has[start+j*stride] for j in range(1, 8)])]
        roster[keep] += 1
        past_rows.append((keep, use))
    next_free = 0
    picks = []
    for s in np.flatnonzero(roster >= 2):
        if s >= next_free:
            picks.append(int(s))
            next_free = s+19*stride+1
    take = np.asarray(picks, dtype=np.int64)
    actual = np.zeros((len(take), 12), dtype=np.int64)
    full = np.zeros(len(take), dtype=np.int64)
    indices = take[:, None]+stride*np.arange(8, 20)
    for starts, use in past_rows:
        good = np.flatnonzero(np.isin(take, starts))
        if not len(good):
            continue
        seen = np.isin(indices[good], use)
        actual[good] += seen
        full[good] += seen.all(1)
    n = roster[take]
    pairs = n*(n-1)//2
    visible = actual*(actual-1)//2
    return dict(stride_ticks=stride, first_time_seconds=lo*TICK, last_time_seconds=hi*TICK,
                available_multiagent_starts=int((roster >= 2).sum()), past_selected_queries=len(take),
                selected_agents=int(n.sum()), agents_with_all_12_timestamp_and_position_tokens=int(full.sum()),
                selected_queries_all_agents_complete=int((full == n).sum()),
                selected_queries_at_least_two_complete=int((full >= 2).sum()),
                selected_pair_steps=int(12*pairs.sum()), observed_pair_step_tokens=int(visible.sum()),
                roster_median=float(np.median(n)) if len(n) else None)
