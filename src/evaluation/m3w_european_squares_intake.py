"""Raw tracked boxes and past-only support; no smoothing, outcomes or model fit."""
from collections import Counter
import hashlib
from pathlib import PurePosixPath
import stat

import numpy as np
import pandas as pd

DTYPE = np.dtype([(name, '<i8' if name in ('agent', 'frame', 'class_id') else '<f8')
                  for name in ('agent', 'frame', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'confidence')])
COLUMNS = ('x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'confidence', 'tracker_id', 'class_name', 'frame_index')


def safe_zip_member(info):
    p = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if (p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(mode)
            or info.flag_bits & 1 or info.file_size < 0 or info.file_size > 5_000_000_000):
        raise ValueError('Unsafe or oversized ZIP member')
    return str(p)


def read_raw_csv(stream, chunksize=250000):
    pieces, names = [], Counter()
    previous_frame = -1
    for chunk in pd.read_csv(stream, chunksize=chunksize):
        if set(chunk.columns) != set(COLUMNS) or len(chunk.columns) != len(COLUMNS):
            raise ValueError('Unexpected raw tracker schema')
        floats = chunk.drop(columns='class_name').to_numpy(dtype=np.float64)
        if not np.isfinite(floats).all():
            raise ValueError('Nonfinite raw field')
        result = np.empty(len(chunk), dtype=DTYPE)
        for target, source in (('agent', 'tracker_id'), ('frame', 'frame_index'), ('class_id', 'class_id')):
            values = chunk[source].to_numpy(dtype=np.float64)
            if (np.any(values < 0) or np.any(values > 2**53-1) or np.any(values != np.floor(values))):
                raise ValueError('Invalid raw integer')
            result[target] = values.astype(np.int64)
        for name in ('x_min', 'y_min', 'x_max', 'y_max', 'confidence'):
            result[name] = chunk[name].to_numpy(dtype=np.float64)
        if (np.any(result['x_max'] < result['x_min']) or np.any(result['y_max'] < result['y_min'])
                or np.any(result['confidence'] < 0) or np.any(result['confidence'] > 1)):
            raise ValueError('Invalid raw box or confidence')
        if len(result) and (result['frame'][0] < previous_frame or np.any(np.diff(result['frame']) < 0)):
            raise ValueError('Raw file not in frame order')
        if len(result):
            previous_frame = int(result['frame'][-1])
        names.update(map(str, chunk['class_name']))
        pieces.append(result)
    if not pieces or not sum(map(len, pieces)):
        raise ValueError('Empty raw recording')
    rows = np.concatenate(pieces)
    rows.sort(order=['agent', 'frame'])
    if np.any((rows['agent'][1:] == rows['agent'][:-1]) & (rows['frame'][1:] == rows['frame'][:-1])):
        raise ValueError('Duplicate scoped agent/frame; recording quarantined, not silently deduplicated')
    return rows, dict(names)


def frame_support(frames, length, stride, future=12):
    """Count exact support along residue chains, including labels beyond short gaps."""
    frames = np.asarray(frames, dtype=np.int64)
    if length < 2 or stride < 1 or future < 1 or np.any(np.diff(frames) <= 0):
        raise ValueError('Invalid support request or repeated/unordered frames')
    count = dict(past_eligible=0, complete_future12=0, partial_future12=0, no_future12=0)
    queries = []
    for phase in np.unique(frames % stride):
        f = frames[frames % stride == phase]
        ends = np.r_[0, np.flatnonzero(np.diff(f) != stride)+1, len(f)]
        for start, end in zip(ends[:-1], ends[1:]):
            n = int(end-start)
            if n < length:
                continue
            q = f[start+length-1:end]
            queries.append(q)
            count['past_eligible'] += n-length+1
            count['complete_future12'] += max(0, n-length-future+1)
            if end == len(f) or f[end]-f[end-1] > future*stride:
                count['no_future12'] += 1
    count['partial_future12'] = count['past_eligible']-count['complete_future12']-count['no_future12']
    return count, np.sort(np.concatenate(queries)) if queries else np.empty(0, dtype=np.int64)


def summarize(rows, class_names, lengths=(8, 16, 32, 64), strides=(1, 12)):
    agents, starts, sizes = np.unique(rows['agent'], return_index=True, return_counts=True)
    gaps = Counter()
    per_frame = Counter(map(int, rows['frame']))
    support = {f'K{k}_stride{s}': dict(past_eligible=0, complete_future12=0, partial_future12=0,
                                      no_future12=0, query_frames=0, query_frames_at_least_agents={})
               for k in lengths for s in strides}
    qcounts = {key: Counter() for key in support}
    track_hashes = []
    for agent, start, size in zip(agents, starts, sizes):
        track = rows[start:start+size]
        frames = track['frame']
        gaps.update(map(int, np.diff(frames)))
        geometry = np.column_stack([frames-frames[0], *[track[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')]])
        track_hashes.append(hashlib.sha256(geometry.astype('<f8').tobytes()).hexdigest())
        for length in lengths:
            for stride in strides:
                key = f'K{length}_stride{stride}'
                counts, queries = frame_support(frames, length, stride)
                for field, value in counts.items():
                    support[key][field] += value
                qcounts[key].update(map(int, queries))
    for key, counts in qcounts.items():
        support[key]['query_frames'] = len(counts)
        support[key]['query_frames_at_least_agents'] = {str(n): sum(v >= n for v in counts.values())
                                                       for n in (2, 5, 10)}
    classes, class_counts = np.unique(rows['class_id'], return_counts=True)
    result = dict(rows=len(rows), tracks=len(agents), frame_min=int(rows['frame'].min()),
                  frame_max=int(rows['frame'].max()), frames_with_agents=len(per_frame),
                  max_simultaneous_agents=max(per_frame.values()), class_name_counts=class_names,
                  class_id_counts={str(k): int(v) for k, v in zip(classes, class_counts)},
                  track_length_quantiles=dict(zip(('min', 'median', 'p95', 'max'),
                                                 map(float, np.quantile(sizes, [0, .5, .95, 1])))),
                  frame_delta_counts={str(k): v for k, v in sorted(gaps.items())},
                  tracks_shorter_than_30=int((sizes < 30).sum()), support=support,
                  duplicate_full_relative_track_geometries=len(track_hashes)-len(set(track_hashes)),
                  rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    return result, track_hashes


def past_scene(rows, query, length=8, stride=12):
    if length < 2 or stride < 1:
        raise ValueError('Invalid observation grid')
    prefix = rows[rows['frame'] <= query]
    agents = np.unique(prefix['agent'][prefix['frame'] == query])
    grid = query - np.arange(length-1, -1, -1)*stride
    output = {}
    for agent in agents:
        track = prefix[prefix['agent'] == agent]
        track = np.sort(track, order='frame')
        if np.any(np.diff(track['frame']) <= 0):
            raise ValueError('Repeated raw agent/frame')
        idx = np.searchsorted(track['frame'], grid)
        valid = idx < len(track)
        valid[valid] &= track['frame'][idx[valid]] == grid[valid]
        boxes = np.zeros((length, 4))
        for i, name in enumerate(('x_min', 'y_min', 'x_max', 'y_max')):
            boxes[valid, i] = track[name][idx[valid]]
        xy = (boxes[:, :2]+boxes[:, 2:])/2
        velocity = np.zeros_like(xy)
        velocity_valid = np.r_[False, valid[1:] & valid[:-1]]
        velocity[1:] = np.where(velocity_valid[1:, None], np.diff(xy, axis=0)/stride, 0)
        output[int(agent)] = dict(frames=grid, boxes=boxes, box_center=xy, valid=valid,
                                 velocity_causal_fd=velocity, velocity_valid=velocity_valid)
    return output
