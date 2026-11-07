"""Bounded causal query packets and resumable prediction chunks, without a label API."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np

from src.data_unification.m3w_dronecrowd_frozen_inputs import adapt_past
from src.evaluation.m3w_dronecrowd_windows import PROBES, past_inputs


def native_prefix(record, frame):
    """Slice time first so advanced indexing cannot copy any later coordinates."""
    if type(frame) is not int or not 7 <= frame < record.valid.shape[1]:
        raise ValueError('Native observation-8 frame is out of bounds')
    time = slice(frame-7, frame+1)
    prefix = replace(record, positions=record.positions[:, time], valid=record.valid[:, time],
                     in_bounds=record.in_bounds[:, time])
    inputs = past_inputs(prefix, 7, PROBES[0])
    inputs.update(query_frame=frame, observation_frames=np.arange(frame-7, frame+1))
    return inputs


def scene_packets(record, frames, pack):
    """Produce ordered prediction rows, plus visibility and zero-target accounting."""
    last = -1
    for frame in frames:
        frame = int(frame)
        if frame <= last:
            raise ValueError('Query frames must be unique and ordered')
        last = frame
        causal = adapt_past(native_prefix(record, frame))
        features, indices = pack(causal)
        ids = causal['agent_id'][indices]
        yield dict(frame=frame, visible=len(causal['agent_id']), eligible=len(ids), geometry=features,
                   keys=np.column_stack((np.full(len(ids), frame, dtype=np.int64), ids)),
                   origin=causal['history_xy'][indices, -1].copy())


def fixed_batches(packets, rows=128):
    """Fill fixed inference batches across adjacent queries without retaining a recording."""
    if rows < 1:
        raise ValueError('Positive batch size required')
    fields = ('keys', 'origin', 'geometry')
    pieces = {k: [] for k in fields}; pending = 0
    for packet in packets:
        start = 0
        while start < packet['eligible']:
            size = min(rows-pending, packet['eligible']-start)
            for k in fields:
                pieces[k].append(packet[k][start:start+size])
            pending += size; start += size
            if pending == rows:
                yield {k: np.concatenate(pieces[k]) for k in fields}
                pieces = {k: [] for k in fields}; pending = 0
    if pending:
        yield {k: np.concatenate(pieces[k]) for k in fields}


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode())
    h.update(a.data)
    return h.hexdigest()


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    pending = path.with_suffix('.json.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    pending.replace(path)


def spool_batch(directory, index, identity, keys, features, infer):
    """A caller holds the run lock; only a verified marker makes a batch resumable."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    payload, mark = (directory/f'{index:06d}.{ext}' for ext in ('npz', 'json'))
    if len(keys) != len(features) or len(keys) == 0 or keys.dtype != np.int64:
        raise ValueError('Nonempty aligned integer prediction keys required')
    request = dict(identity=identity, index=index, keys_sha256=array_hash(keys),
                   geometry_sha256=array_hash(features), rows=len(keys))
    if mark.exists():
        saved = json.loads(mark.read_text())
        if (saved['request'] != request or not payload.exists()
                or file_hash(payload) != saved['payload_sha256']):
            raise ValueError('Completed prediction chunk does not match the frozen request')
        with np.load(payload, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['keys'], keys)
            return z['prediction'], True
    prediction = np.asarray(infer(features))
    if (prediction.shape != (len(keys), 12, 2) or prediction.dtype != np.float32
            or not np.isfinite(prediction).all()):
        raise ValueError('Expected finite float32 native 12-step forecasts')
    pending = payload.with_suffix('.npz.pending')
    with pending.open('wb') as f:
        np.savez(f, keys=keys, prediction=prediction)
    pending.replace(payload)
    write_json(mark, dict(request=request, payload_sha256=file_hash(payload)))
    return prediction, False
