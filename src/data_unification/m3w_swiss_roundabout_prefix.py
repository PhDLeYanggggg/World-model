"""Decode only the planned observed positions of each Swiss trajectory query."""

from bisect import bisect_right
import csv

import numpy as np

from src.data_unification.m3w_pneuma_prefix import ecef
from src.data_unification.m3w_swiss_roundabout_support import clock_token, roster


# Fixed before numeric parsing; this is a map, not a fit to held trajectories.
ORIGIN = (47.5, 8.1)
HISTORY = 8


def project(lat_lon):
    lat, lon = np.deg2rad(ORIGIN)
    axes = np.array([[-np.sin(lon), np.cos(lon), 0.],
                     [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)]])
    return (ecef(lat_lon)-ecef(ORIGIN)) @ axes.T


def plan(segments, stride):
    required = {}
    offsets = [0]
    agents, segment_ids, end_ticks = [], [], []
    for segment_id, s in enumerate(segments):
        ids = sorted(s['tracks'])
        ends, membership = roster([s['tracks'][i] for i in ids], s['ticks'], stride)
        by_query = [[] for _ in ends]
        for agent, rows in zip(ids, membership, strict=True):
            for i in rows:
                by_query[i].append(agent)
        for end, who in zip(ends, by_query, strict=True):
            if len(who) < 2:
                raise ValueError('Inconsistent past roster')
            for agent in who:
                row = len(agents)
                for step in range(HISTORY):
                    key = (segment_id, agent, int(end)-(HISTORY-1-step)*stride)
                    if key in required:
                        raise ValueError('Overlapping observed prefixes')
                    required[key] = (row, step)
                agents.append(agent)
            offsets.append(len(agents))
            segment_ids.append(segment_id)
            end_ticks.append(int(end))
    return required, dict(offsets=np.array(offsets, dtype=np.int64), agent_id=np.array(agents, dtype=np.int64),
                          clock_segment=np.array(segment_ids, dtype=np.int32), current_tick=np.array(end_ticks, dtype=np.int64))


def pack(stream, segments, required, meta):
    starts = [s['first_clock_centisecond'] for s in segments]
    ends = [s['last_clock_centisecond'] for s in segments]
    reader = csv.DictReader(stream)
    if reader.fieldnames != ['track_id', 'type', 'lon', 'lat', 'time']:
        raise ValueError('Unexpected release columns')
    missing = dict(required)
    pos = np.empty((len(meta['agent_id']), HISTORY, 2))
    for row in reader:
        clock = clock_token(row['time'])
        segment = bisect_right(starts, clock)-1
        if segment < 0 or clock > ends[segment]:
            continue
        tick = ((clock-starts[segment])*3+5)//10
        key = (segment, int(row['track_id']), tick)
        at = missing.pop(key, None)
        if at is not None:
            pos[at] = (float(row['lat']), float(row['lon']))
    if missing:
        raise ValueError('Audited past positions are missing')
    history = project(pos)
    offsets = meta['offsets']
    origin = history[offsets[:-1], -1].copy()
    history -= np.repeat(origin, np.diff(offsets), axis=0)[:, None, :]
    return dict(meta, history=history, origin=origin)
