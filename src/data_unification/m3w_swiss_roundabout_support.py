"""Audit the released Swiss vehicle clock and past-defined label support."""

from collections import Counter, defaultdict
import csv
import re

import numpy as np


FILE = re.compile(r'D([12])_([AP]M)([1-7])_(F[12]|L1)\.csv')
CLOCK = re.compile(r'([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{2})(?:0000)?')
KINDS = {'Car', 'Bus', 'Truck'}


def clock_token(value):
    m = CLOCK.fullmatch(value)
    if not m:
        raise ValueError('Invalid or non-centisecond clock')
    h, m, s, centi = map(int, m.groups())
    return (h*3600+m*60+s)*100+centi


def grid(stamps, rate):
    delta = stamps-stamps[0]
    ticks = np.rint(delta*rate/100.).astype(np.int64)
    residual = delta*10000.-ticks*(1e6/rate)
    return ticks, float(np.ptp(residual))


def scan(stream):
    reader = csv.DictReader(stream)
    if reader.fieldnames != ['track_id', 'type', 'lon', 'lat', 'time']:
        raise ValueError('Unexpected publisher columns')
    times, class_sets = defaultdict(list), defaultdict(set)
    blanks, rows = 0, 0
    for row in reader:
        if None in row or any(v is None for v in row.values()) or row['type'] not in KINDS:
            raise ValueError('Malformed trajectory row')
        track = int(row['track_id'])
        class_sets[track].add(row['type'])
        t = clock_token(row['time'])
        rows += 1
        if row['lon'] and row['lat']:
            times[track].append(t)
        else:
            blanks += 1
    if not times:
        raise ValueError('No observed coordinate tokens')
    times = {k: np.asarray(v, dtype=np.int64) for k, v in times.items()}
    if any(not (np.diff(t) > 0).all() for t in times.values()):
        raise ValueError('Duplicate or unordered track ticks')
    raw_times = np.unique(np.concatenate(list(times.values())))
    _, width = grid(raw_times, 30.)
    _, nominal_width = grid(raw_times, 30000/1001)
    segments = []
    for stamp in np.split(raw_times, np.flatnonzero(np.diff(raw_times) > 4)+1):
        ticks, local_width = grid(stamp, 30.)
        if local_width > 10000.01 or not (np.diff(ticks) == 1).all():
            raise ValueError('An uninterrupted clock segment does not fit 30 Hz')
        tracks = {}
        for agent, t in times.items():
            i, j = np.searchsorted(t, [stamp[0], stamp[-1]], side='left')
            if j < len(t) and t[j] == stamp[-1]:
                j += 1
            if j > i:
                tracks[agent] = ticks[np.searchsorted(stamp, t[i:j])]
        segments.append(dict(first_clock_centisecond=int(stamp[0]), last_clock_centisecond=int(stamp[-1]),
                             residual_width_us=local_width, ticks=len(ticks), tracks=tracks))
    classes = Counter(next(iter(v)) for v in class_sets.values() if len(v) == 1)
    stats = dict(rows=rows, coordinate_token_blank_rows=blanks, tracks=len(class_sets), track_types=dict(classes),
                 ambiguous_track_types=sum(len(v) != 1 for v in class_sets.values()), observed_grid_ticks=len(raw_times),
                 first_clock_centisecond=int(raw_times[0]),
                 last_clock_centisecond=int(raw_times[-1]), release_grid_residual_width_us=width,
                 nominal_ntsc_grid_residual_width_us=nominal_width,
                 clock_segments=[{k: v for k, v in s.items() if k != 'tracks'} for s in segments],
                 track_length_p10_p50_p90=np.quantile([len(t) for t in times.values()], [.1, .5, .9]).tolist())
    return segments, stats


def history_ends(track, stride, history=8):
    """Return current ticks with the required observations, independent of future."""
    if len(track) < history:
        return np.empty(0, dtype=np.int64)
    out = []
    for phase in np.unique(track % stride):
        times = track[track % stride == phase]
        cuts = np.r_[0, np.flatnonzero(np.diff(times) != stride)+1, len(times)]
        out.extend(times[a+history-1:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b-a >= history)
    return np.sort(np.concatenate(out)) if out else np.empty(0, dtype=np.int64)


def roster(tracks, ticks, stride):
    if stride < 1:
        raise ValueError('Require a positive integer stride')
    ends = [history_ends(track, stride) for track in tracks]
    count = np.zeros(ticks, dtype=np.int32)
    for e in ends:
        count[e] += 1
    picked = []
    for e in np.flatnonzero(count[:max(0, ticks-12*stride)] >= 2):
        if not picked or e-picked[-1] >= 20*stride:
            picked.append(e)
    selected = np.asarray(picked, dtype=np.int64)
    return selected, [np.flatnonzero(np.isin(selected, e)) for e in ends]


def coverage(tracks, ticks, stride):
    selected, membership = roster(tracks, ticks, stride)
    targets = np.zeros(len(selected), dtype=np.int32)
    complete = np.zeros_like(targets)
    pair_obs, pair_all = 0, 0
    totals = Counter()
    per_step = np.zeros((len(selected), 12), dtype=np.int32)
    for track, rows in zip(tracks, membership, strict=True):
        future_ticks = selected[rows, None]+stride*np.arange(1, 13)
        present = np.isin(future_ticks, track)
        n = present.sum(axis=1)
        targets[rows] += 1
        complete[rows] += n == 12
        per_step[rows] += present
        totals.update(complete=int((n == 12).sum()), partial=int(((n > 0) & (n < 12)).sum()),
                      absent=int((n == 0).sum()), endpoints=int(present[:, -1].sum()))
    pair_obs = int(np.sum(per_step*(per_step-1)//2))
    pair_all = int(12*np.sum(targets*(targets-1)//2))
    return dict(queries=len(selected), target_rows=int(targets.sum()), **totals,
                queries_all_targets_complete=int((complete == targets).sum()),
                queries_two_complete=int((complete >= 2).sum()),
                pair_step_slots=pair_all, observed_pair_step_slots=pair_obs)
