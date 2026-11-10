"""Causal pNEUMA pilot prefixes and a fixed WGS84 topocentric map."""

import csv

import numpy as np

from src.data_unification.m3w_pneuma_support import ABSENT, HEADER, TICK


ORIGIN_LAT_LON = (38., 23.75)
HISTORY, FUTURE = 8, 12


def ecef(lat_lon):
    x = np.asarray(lat_lon, dtype=float)
    if x.shape[-1:] != (2,) or not np.isfinite(x).all() or (np.abs(x) > [90., 180.]).any():
        raise ValueError('Expected finite WGS84 latitude and longitude in degrees')
    lat, lon = np.moveaxis(np.deg2rad(x), -1, 0)
    f = 1./298.257223563
    e2 = f*(2.-f)
    n = 6378137./np.sqrt(1.-e2*np.sin(lat)**2)
    return np.stack((n*np.cos(lat)*np.cos(lon), n*np.cos(lat)*np.sin(lon), n*(1.-e2)*np.sin(lat)), axis=-1)


def east_north(lat_lon):
    lat, lon = np.deg2rad(ORIGIN_LAT_LON)
    axes = np.array([[-np.sin(lon), np.cos(lon), 0.],
                     [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)]])
    return (ecef(lat_lon)-ecef(ORIGIN_LAT_LON)) @ axes.T


def tracks(stream):
    source = csv.reader(stream, delimiter=';')
    header = tuple(c.strip().lower() for c in next(source))
    if header[:len(HEADER)] != HEADER or any(header[len(HEADER):]):
        raise ValueError('Unexpected pNEUMA columns')
    seen = set()
    for row in source:
        if not row or not any(c.strip() for c in row):
            continue
        if (len(row)-4) % 6 == 1 and not row[-1].strip():
            row = row[:-1]
        if len(row) < 10 or (len(row)-4) % 6:
            raise ValueError('Incomplete pNEUMA track row')
        agent = int(row[0])
        kind = row[1].strip()
        if agent in seen or not kind:
            raise ValueError('Duplicate track ID or absent type')
        seen.add(agent)
        t = np.asarray(row[9::6], dtype=float)
        if not np.isfinite(t).all() or (t < 0.).any() or (np.diff(t) <= 0.).any():
            raise ValueError('Invalid file-local clock')
        clock = np.rint(t/TICK).astype(np.int64)
        on_grid = np.abs(t-clock*TICK) <= 1e-6
        ix = np.flatnonzero(on_grid)
        if (np.diff(clock[ix]) <= 0).any():
            raise ValueError('Duplicate grid tick')
        yield agent, kind, clock[ix], ix, row


def plan(stream, stride):
    if type(stride) is not int or stride < 1:
        raise ValueError('Expected positive tick stride')
    times = {}
    lo, hi = None, None
    for agent, _, clock, ix, row in tracks(stream):
        if not len(clock):
            continue
        lo = int(clock[0]) if lo is None else min(lo, int(clock[0]))
        hi = int(clock[-1]) if hi is None else max(hi, int(clock[-1]))
        present = np.fromiter((row[4+6*j].strip().lower() not in ABSENT and row[5+6*j].strip().lower() not in ABSENT
                               for j in ix), dtype=bool, count=len(ix))
        times[agent] = clock[present]
    if lo is None or not 1 <= hi-lo+1 <= 100_000:
        raise ValueError('Missing or unexpectedly long release clock')
    horizon = (HISTORY+FUTURE-1)*stride
    counts = np.zeros(max(0, hi-lo+1-horizon), dtype=np.int64)
    past = {}
    for agent, a in times.items():
        steps = a-lo
        has = np.zeros(hi-lo+1, dtype=bool)
        has[steps] = True
        start = steps[steps < len(counts)]
        start = start[np.logical_and.reduce([has[start+j*stride] for j in range(1, HISTORY)])]
        past[agent] = start
        counts[start] += 1
    picks, free = [], 0
    for s in np.flatnonzero(counts >= 2):
        if s >= free:
            picks.append(int(s))
            free = s+horizon+1
    picks = np.asarray(picks, dtype=np.int64)
    starts = {agent: picks[np.isin(picks, a)]+lo for agent, a in past.items()}
    starts = {agent: a for agent, a in starts.items() if len(a)}
    return starts, picks+lo, counts[picks]


def pack(stream, wanted, starts, counts, stride):
    history, agent_id, kind, tick, begin = [], [], [], [], []
    remain = set(wanted)
    for agent, name, clock, ix, row in tracks(stream):
        if agent not in wanted:
            continue
        start = wanted[agent]
        grid = start[:, None]+stride*np.arange(HISTORY)
        found = np.searchsorted(clock, grid)
        if (found >= len(clock)).any() or not np.array_equal(clock[found], grid):
            raise ValueError('Audited past grid no longer matches')
        cols = 4+6*ix[found].ravel()
        pos = np.array([(float(row[j]), float(row[j+1])) for j in cols]).reshape(-1, HISTORY, 2)
        history.append(east_north(pos))
        agent_id.extend([agent]*len(start))
        kind.extend([name]*len(start))
        tick.append(grid)
        begin.extend(start.tolist())
        remain.remove(agent)
    if remain or not history:
        raise ValueError('Missing audited past track')
    agent, time = np.asarray(agent_id), np.asarray(begin)
    order = np.lexsort((agent, time))
    history = np.concatenate(history)[order]
    offset = np.r_[0, counts.cumsum()]
    if not np.array_equal(time[order], np.repeat(starts, counts)):
        raise ValueError('Past roster differs from plan')
    origin = history[offset[:-1], -1].copy()
    history -= np.repeat(origin, counts, axis=0)[:, None, :]
    return dict(history=history, offsets=offset, origin=origin, start_tick=starts,
                past_tick=np.concatenate(tick)[order], agent_id=agent[order], vehicle_type=np.asarray(kind)[order])
