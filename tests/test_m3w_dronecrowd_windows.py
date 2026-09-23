"""Past/future separation tests, not proof of annotation-time provenance."""
from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.m3w_dronecrowd_windows import (
    WindowSpec, audit_support, compare_inputs, parse_recording,
    past_inputs, target_labels,
)


def fixture_xml(n=30, hidden=None, births=None):
    hidden, births = hidden or {}, births or {}
    tracks = []
    for agent in range(3):
        boxes = []
        for frame in range(n):
            outside = frame < births.get(agent, 0) or frame in hidden.get(agent, ())
            x = 10 + agent * 10 + frame
            boxes.append(f'<box frame="{frame}" xtl="{x}" ytl="2" xbr="{x+4}" '
                         f'ybr="6" outside="{int(outside)}" occluded="0"/>')
        tracks.append(f'<track id="{agent}" label="human">{"".join(boxes)}</track>')
    return f'<annotations count="3">{"".join(tracks)}</annotations>'.encode()


def test_future_disappearance_changes_labels_not_inputs_or_neighbors():
    spec = WindowSpec("main", 8, 1, tuple(range(1, 13)))
    complete = parse_recording(fixture_xml(), "00001")
    disappears = parse_recording(fixture_xml(hidden={1: range(8, 30)}), "00001")
    left, right = past_inputs(complete, 7, spec), past_inputs(disappears, 7, spec)
    assert compare_inputs(left, right)
    assert right["agent_ids"].tolist() == [0, 1, 2]
    assert target_labels(complete, left, spec)["complete_target"].tolist() == [True] * 3
    assert target_labels(disappears, right, spec)["complete_target"].tolist() == [True, False, True]


def test_future_only_agent_is_never_an_input():
    record = parse_recording(fixture_xml(births={2: 8}), "00001")
    inputs = past_inputs(record, 7, WindowSpec("main", 8, 1, (1,)))
    assert inputs["agent_ids"].tolist() == [0, 1]
    assert inputs["positions"].shape == (2, 8, 2)


def test_future_coordinates_and_track_inventory_cannot_change_prefix():
    record = parse_recording(fixture_xml(births={2: 8}), "00001")
    spec = WindowSpec("main", 8, 1, (1,))
    original = past_inputs(record, 7, spec)
    changed = record.positions.copy()
    changed[:, 8:] = 100000
    short = replace(record, positions=changed[:2, :8], valid=record.valid[:2, :8],
                    agent_ids=record.agent_ids[:2], in_bounds=record.in_bounds[:2, :8])
    assert compare_inputs(original, past_inputs(short, 7, spec))


def test_gap_and_birth_are_masked_not_interpolated_or_future_filtered():
    record = parse_recording(fixture_xml(hidden={1: [3]}, births={2: 7}), "00001")
    spec = WindowSpec("main", 8, 1, (1, 2))
    inputs = past_inputs(record, 7, spec)
    assert inputs["agent_ids"].tolist() == [0, 1, 2]
    assert inputs["history_complete"].tolist() == [True, False, False]
    assert inputs["history_mask"][2].tolist() == [False] * 7 + [True]
    assert not inputs["positions"][1, 3].any()
    assert not inputs["velocity_mask"][1, 3:5].any()
    assert not inputs["velocity_mask"][2].any()
    assert target_labels(record, inputs, spec)["complete_target"].tolist() == [True, False, False]


def test_backward_difference_has_raw_step_units_and_no_initial_velocity():
    record = parse_recording(fixture_xml(), "00001")
    inputs = past_inputs(record, 14, WindowSpec("stride2", 8, 2, (2,)))
    np.testing.assert_array_equal(inputs["velocities"][:, 1:, 0], 1)
    assert not inputs["velocity_mask"][:, 0].any()
    assert not inputs["velocities"][:, 0].any()


def test_stride_does_not_bridge_unsampled_visibility_gap():
    record = parse_recording(fixture_xml(hidden={1: [3]}), "00001")
    spec = WindowSpec("stride2", 8, 2, (2,))
    inputs = past_inputs(record, 14, spec)
    assert inputs["history_mask"][1].all()
    assert not inputs["history_complete"][1]
    assert not inputs["velocity_mask"][1, 2]


def test_future_gap_between_sampled_targets_invalidates_complete_label():
    record = parse_recording(fixture_xml(hidden={1: [9]}), "00001")
    spec = WindowSpec("target_stride", 8, 1, (2, 4))
    inputs = past_inputs(record, 7, spec)
    labels = target_labels(record, inputs, spec)
    assert labels["future_mask"][1].tolist() == [False, True]
    assert not labels["complete_target"][1]
    spec = WindowSpec("endpoint", 8, 1, (4,))
    labels = target_labels(record, inputs, spec)
    assert labels["future_mask"][1].all()
    assert not labels["complete_target"][1]


def test_out_of_bounds_are_flagged_not_clipped_or_auto_excluded():
    raw = fixture_xml().replace(b'xtl="10"', b'xtl="-10"')
    record = parse_recording(raw, "00001")
    inputs = past_inputs(record, 7, WindowSpec("main", 8, 1, (1,)))
    assert not inputs["in_bounds_mask"][0, 0]
    assert inputs["positions"][0, 0, 0] == 2
    assert inputs["history_mask"][0, 0]


@pytest.mark.parametrize("spec", [
    WindowSpec("bad", 0, 1, (1,)), WindowSpec("bad", 8, 0, (1,)),
    WindowSpec("bad", 8, 1, (0,)), WindowSpec("bad", 8, 1, (2, 1)),
    WindowSpec("bad", 8, 1, (1, 1)), WindowSpec("bad", 8, 1, ()),
])
def test_bad_window_specs_refused(spec):
    with pytest.raises(ValueError):
        past_inputs(parse_recording(fixture_xml(), "00001"), 7, spec)


def test_insufficient_past_and_future_fail_explicitly():
    record = parse_recording(fixture_xml(), "00001")
    spec = WindowSpec("main", 8, 1, tuple(range(1, 13)))
    with pytest.raises(ValueError, match="history"):
        past_inputs(record, 6, spec)
    inputs = past_inputs(record, 29, spec)
    with pytest.raises(ValueError, match="future"):
        target_labels(record, inputs, spec)


def test_support_separates_scene_queries_targets_and_future_filtered_neighbors():
    record = parse_recording(fixture_xml(n=20, hidden={1: range(8, 20)}), "00001")
    result = audit_support(record, WindowSpec("main", 8, 1, tuple(range(1, 13))))
    assert result["query_frames_considered"] == result["queries_with_targets"] == 1
    assert result["target_agent_windows"] == 2
    assert result["past_visible_agent_instances"] == 3
    assert result["history_complete_agent_instances"] == 3
    assert result["history_complete_but_future_incomplete_instances"] == 1
    assert result["queries_losing_history_complete_neighbors_if_target_filtered"] == 1
    assert result["target_count_histogram"] == {"2": 1}
    assert result["nonoverlapping_raw_intervals_with_targets"] == 1


def test_nonoverlap_count_is_temporal_not_independent_scene_count():
    record = parse_recording(fixture_xml(n=40), "00001")
    result = audit_support(record, WindowSpec("main", 8, 1, tuple(range(1, 13))))
    assert result["queries_with_targets"] == 21
    assert result["target_agent_windows"] == 63
    assert result["nonoverlapping_raw_intervals_with_targets"] == 2


def test_unknown_geometry_source_or_non_human_refused():
    with pytest.raises(ValueError):
        parse_recording(b"frame,agent,x,y", "00001")
    with pytest.raises(ValueError, match="human"):
        parse_recording(fixture_xml().replace(b'human', b'other'), "00001")
