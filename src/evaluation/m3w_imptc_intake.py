"""Strict IMPTC source parsing and past-indexed support, without predictions."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import PurePosixPath

import numpy as np

ROW_DTYPE = np.dtype([('agent_id', '<i8'), ('frame_id', '<i8'), ('timestamp', '<i8'),
                      ('x', '<f8'), ('y', '<f8'), ('z', '<f8')])


def strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=bad)


def integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a source integer')
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid source integer') from exc
    if not number.is_finite() or number != number.to_integral_value() or abs(number) > 2**63-1:
        raise ValueError('Non-integral or out-of-range source integer')
    return int(number)


def safe_member(member):
    path = PurePosixPath(member.name)
    if (path.is_absolute() or '..' in path.parts or '\\' in member.name
            or not (member.isdir() or member.isreg()) or member.issparse()
            or member.size < 0 or member.size > 400_000_000):
        raise ValueError('Unsafe or excessive source archive member')
    return str(path)


def parse_master(payload):
    source = strict_json(payload)
    if not isinstance(source, dict) or not source:
        raise ValueError('Nonempty master timestamp mapping required')
    entries = sorted((integer(ts), integer(row['id'])) for ts, row in source.items())
    result = np.array(entries, dtype=np.int64)
    if np.any(np.diff(result[:, 0]) <= 0) or np.any(np.diff(result[:, 1]) <= 0):
        raise ValueError('Master timestamp and index must be strictly increasing')
    return result


def parse_track(payload, master, agent_id):
    source = strict_json(payload)
    observations = source['track_data']
    lookup = dict(map(tuple, master.tolist()))
    rows, status, sensor = [], Counter(), Counter()
    for key, observation in observations.items():
        # The release uses ordinal keys, unlike the timestamp-key documentation.
        integer(key)
        ts = integer(observation['ts'])
        if ts not in lookup:
            raise ValueError('Track timestamp not exactly joined to source master')
        xyz = np.asarray(observation['coordinates'], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Finite XYZ triplet required')
        rows.append((integer(agent_id), lookup[ts], ts, *xyz))
        status[str(observation.get('status', 'not_provided'))] += 1
        sensor[str(observation.get('source_type', 'not_provided'))] += 1
    if not rows:
        raise ValueError('Empty track')
    result = np.array(rows, dtype=ROW_DTYPE)
    result.sort(order='frame_id')
    if np.any(np.diff(result['frame_id']) <= 0):
        raise ValueError('Duplicate frame within track')
    # Overview and status are audit labels, never eligibility or inference features.
    overview = source.get('overview', {})
    audit = dict(rows=len(rows), class_id=overview.get('class_id'),
        class_name=overview.get('class_name'), status_counts=dict(status), sensor_counts=dict(sensor),
        overview_length=overview.get('length', overview.get('lenght')), first_frame=int(result['frame_id'][0]),
        last_frame=int(result['frame_id'][-1]), gap_count=int((np.diff(result['frame_id']) != 1).sum()))
    return result, audit


def past_window(rows, agent_id, query, *, length=8, stride=1):
    if type(length) is not int or type(stride) is not int or length < 2 or stride < 1:
        raise ValueError('At least two observations and positive integer stride required')
    frames = integer(query) - np.arange(length-1, -1, -1) * stride
    prefix = rows[(rows['agent_id'] == agent_id) & (rows['frame_id'] <= query)]
    prefix = np.sort(prefix, order='frame_id')
    if np.any(np.diff(prefix['frame_id']) <= 0):
        raise ValueError('Duplicate frame within track')
    where = np.searchsorted(prefix['frame_id'], frames)
    valid = where < len(prefix)
    valid[valid] &= prefix['frame_id'][where[valid]] == frames[valid]
    xy = np.zeros((length, 2))
    xy[valid] = np.column_stack((prefix['x'], prefix['y']))[where[valid]]
    velocity = np.zeros_like(xy)
    velocity_valid = np.zeros(length, dtype=bool)
    velocity_valid[1:] = valid[1:] & valid[:-1]
    velocity[1:] = np.where(velocity_valid[1:, None], np.diff(xy, axis=0)/stride, 0)
    return dict(frame_id=frames, xy=xy, valid_mask=valid, velocity_causal_fd=velocity,
        velocity_valid_mask=velocity_valid, velocity_unit='source_coordinate_per_master_index')


def past_scene(rows, query, *, length=8, stride=1):
    current = rows[rows['frame_id'] == query]
    if len(np.unique(current['agent_id'])) != len(current):
        raise ValueError('Duplicate agent at query')
    return {int(agent): past_window(rows, int(agent), query, length=length, stride=stride)
            for agent in sorted(current['agent_id'])}


def support(rows, *, lengths=(8, 16, 32, 64), strides=(1, 10)):
    """Query eligibility is past-only; target completeness is counted separately."""
    if len(rows) == 0:
        raise ValueError('No source rows')
    result = {}
    for stride in strides:
        for length in lengths:
            count = dict(past_eligible=0, complete_future12=0, partial_future12=0, no_future12=0)
            per_frame = Counter()
            for agent in np.unique(rows['agent_id']):
                f = np.sort(rows['frame_id'][rows['agent_id'] == agent])
                if len(np.unique(f)) != len(f):
                    raise ValueError('Duplicate source track frame')
                eligible = np.isin(f[:, None] - np.arange(length)*stride, f).all(1)
                q = f[eligible]
                per_frame.update(map(int, q))
                future = np.isin(q[:, None] + np.arange(1, 13)*stride, f)
                count['past_eligible'] += int(eligible.sum())
                count['complete_future12'] += int(future.all(1).sum())
                count['partial_future12'] += int((future.any(1) & ~future.all(1)).sum())
                count['no_future12'] += int((~future.any(1)).sum())
            count['query_frames'] = len(per_frame)
            count['query_frames_at_least_agents'] = {
                str(n): sum(value >= n for value in per_frame.values()) for n in (2, 5, 10)}
            result[f'K{length}_stride{stride}'] = count
    return result
