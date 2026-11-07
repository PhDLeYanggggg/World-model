from dataclasses import replace

import numpy as np
import pytest

from src.data_unification.m3w_causal_forecast_stream import fixed_batches, native_prefix, scene_packets, spool_batch
from src.evaluation.m3w_dronecrowd_windows import PROBES, Recording, compare_inputs, past_inputs
from src.world_model.m3w_european_source_forecast import pack_scene


def recording():
    xy = np.arange(4*30*2, dtype=float).reshape(4, 30, 2)
    valid = np.ones((4, 30), bool); valid[1, 4] = False; valid[:, 10] = False
    return Recording('synthetic', 'sha', np.array([1, 3, 6, 9]), xy, valid, valid.copy())


def test_prefix_matches_original_and_ignores_later_coordinates_and_visibility():
    r = recording(); frame = 8
    xy, valid = r.positions.copy(), r.valid.copy()
    xy[:, frame+1:] = np.nan; valid[:, frame+1:] = ~valid[:, frame+1:]
    expected = past_inputs(r, frame, PROBES[0]); actual = native_prefix(r, frame)
    assert compare_inputs(expected, actual)
    assert compare_inputs(expected, native_prefix(replace(r, positions=xy, valid=valid), frame))


def test_reader_never_indexes_the_full_time_axis_first():
    r = recording(); frame = 8
    class GuardedTimeAxis:
        def __init__(self, arr):
            self.arr = arr
            self.shape = arr.shape

        def __getitem__(self, key):
            assert key == (slice(None), slice(frame-7, frame+1))
            return self.arr[key]

    protected = replace(r, positions=GuardedTimeAxis(r.positions),
                        valid=GuardedTimeAxis(r.valid), in_bounds=GuardedTimeAxis(r.in_bounds))
    assert compare_inputs(past_inputs(r, frame, PROBES[0]), native_prefix(protected, frame))


def test_packets_retain_empty_and_partial_history_queries_and_batch_alignment():
    packets = list(scene_packets(recording(), [8, 9, 10], pack_scene))
    assert [(p['visible'], p['eligible']) for p in packets] == [(4, 3), (4, 3), (0, 0)]
    batches = list(fixed_batches(packets, rows=4))
    assert [len(b['keys']) for b in batches] == [4, 2]
    for field in ('keys', 'origin', 'geometry'):
        np.testing.assert_array_equal(np.concatenate([b[field] for b in batches]),
                                      np.concatenate([p[field] for p in packets]))


def test_spool_reuses_verified_output_and_never_accepts_mismatched_inputs(tmp_path):
    keys = np.arange(4, dtype=np.int64); x = np.zeros((4, 476), np.float32)
    def infer(f): return np.zeros((len(f), 12, 2), np.float32)
    prediction, reused = spool_batch(tmp_path, 0, 'fixture', keys, x, infer)
    assert not reused
    def no_second_inference(_): raise AssertionError('Already committed')
    value, reused = spool_batch(tmp_path, 0, 'fixture', keys, x, no_second_inference)
    assert reused; np.testing.assert_array_equal(value, prediction)
    with pytest.raises(ValueError, match='frozen request'):
        spool_batch(tmp_path, 0, 'fixture', keys, x+1, no_second_inference)
    (tmp_path/'000000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='frozen request'):
        spool_batch(tmp_path, 0, 'fixture', keys, x, no_second_inference)
