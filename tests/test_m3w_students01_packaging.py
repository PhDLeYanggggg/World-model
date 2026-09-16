import numpy as np
import pytest

from scripts.audit_m3w_students01_packaging import inspect_packaging
from src.data_unification.m3w_causal_recordings import RecordingWindows, write_recording


def example():
    full = np.array([[i * 10, agent, i * .1234567 + agent, .9876543]
                     for agent, length in ((1, 45), (2, 12)) for i in range(length)])
    chunks = full[full[:, 1] == 1][:40].copy()
    chunks[20:, 1] = 3
    chunks[:, 2:] = np.round(chunks[:, 2:], 3)
    return full, chunks


def test_fragmentation_and_future_availability_are_not_frame_reset():
    result = inspect_packaging(*example())
    assert result['matched_rows_same_native_frame_and_rounded_xy'] == 40
    assert result['removed_rows'] == 17
    assert result['original_tracks_split_into_multiple_ids'] == 1
    assert result['upstream_future_availability_conditioning']
    assert not result['frame_clock_mismatch_detected']
    assert result['support']['continuous']['past_supported_agent_queries'] > result['support']['packaged']['past_supported_agent_queries']


def test_changed_clock_is_refused():
    full, chunks = example()
    chunks[0, 0] += 1
    with pytest.raises(ValueError, match='uniquely match'):
        inspect_packaging(full, chunks)


def test_future_deletion_changes_packaged_past_eligibility_not_past_positions():
    full, _ = example()
    one = full[full[:, 1] == 1]
    past = one[:8].copy()
    assert len(one) // 20 > 0
    assert len(past) // 20 == 0
    np.testing.assert_array_equal(one[one[:, 0] <= 70], past)


def test_continuous_reader_keeps_short_agents_and_ignores_future_deletion(tmp_path):
    full, _ = example()
    before_path, after_path = tmp_path / 'before', tmp_path / 'after'
    metadata = {'id': 'same', 'physical_scene': 'same'}
    write_recording(before_path, full, metadata)
    write_recording(after_path, full[full[:, 0] <= 70], metadata)
    before, after = RecordingWindows(before_path), RecordingWindows(after_path)
    a, b = [r.get_scene_inputs(70, 120, history_steps=8) for r in (before, after)]
    assert [x['agent_id'] for x in a['agents']] == [1, 2]
    assert [x['agent_id'] for x in b['agents']] == [1, 2]
    for x, y in zip(a['agents'], b['agents']):
        for name, value in x['inputs'].items():
            np.testing.assert_array_equal(value, y['inputs'][name])
    assert all(not x['future_label_mask'].any() for x in after.get_scene_labels(b))
