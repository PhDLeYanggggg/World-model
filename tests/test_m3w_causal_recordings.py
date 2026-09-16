from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.data_unification.m3w_causal_recordings import (
    BASELINES, FEATURE_NAMES, PROTOCOL_RAW, PROTOCOL_STEPS, RecordingWindows,
    build_window_index, causal_baselines, clean_points, validate_catalog,
    validate_split_groups, write_recording,
)
from src.evaluation.m3w_recording_lineage import sha256


def points():
    return np.asarray([[i * 10, agent, i * 0.1 + agent, agent * 0.2]
                       for agent in (1, 2, 3) for i in range(40)], dtype=float)


def store(tmp_path):
    write_recording(tmp_path, points(), {"id": "fixture", "physical_scene": "fixture_scene"})
    return RecordingWindows(tmp_path)


def test_exact_horizons_do_not_snap_to_next_frame():
    p, _ = clean_points(points())
    idx = build_window_index(p)
    assert not ((idx["protocol"] == PROTOCOL_RAW) & (idx["horizon_raw"] == 25)).any()
    assert {10, 50, 100} == set(idx["horizon_raw"][idx["protocol"] == PROTOCOL_RAW])
    np.testing.assert_array_equal(p[idx["future_end"], 0] - p[idx["current_row"], 0], idx["horizon_raw"])


def test_step_horizon_keeps_actual_frame_definition():
    p = points()
    p[:, 0] *= 0.6
    p, _ = clean_points(p)
    idx = build_window_index(p)
    steps = idx[idx["protocol"] == PROTOCOL_STEPS]
    assert len(steps) > 0
    assert set(steps["horizon_raw"]) == {72}
    assert len(idx[idx["protocol"] == PROTOCOL_RAW]) == 0


def test_gaps_are_neither_interpolated_nor_crossed():
    p = points()
    p = p[~((p[:, 1] == 1) & (p[:, 0] == 100))]
    p, _ = clean_points(p)
    idx = build_window_index(p)
    for row in idx:
        span = p[row["history_start"]:row["future_end"] + 1]
        assert np.all(np.diff(span[:, 0]) == 10)
        assert len(set(span[:, 1])) == 1


def test_conflicting_positions_fail_closed():
    p = points()
    p2 = np.concatenate([p, [p[0] + np.array([0, 0, 1, 0])]])
    with pytest.raises(ValueError, match="Conflicting"):
        clean_points(p2)
    cleaned, n = clean_points(np.concatenate([p, p[:2]]))
    assert n == 2 and len(cleaned) == len(p)


def test_future_mutations_do_not_change_any_inference_feature(tmp_path):
    ds = store(tmp_path)
    sample = int(np.flatnonzero((ds.index["protocol"] == PROTOCOL_RAW) & (ds.index["horizon_raw"] == 50))[0])
    before = ds.get_inputs(sample)
    old_labels = ds.get_labels(sample)
    current = ds.identity(sample)["frame_id"]
    changed = np.asarray(ds.points).copy()
    changed[changed[:, 0] > current, 2:] += 10000
    ds.points = changed
    after = ds.get_inputs(sample)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(old_labels["future_xy_dataset_local"], ds.get_labels(sample)["future_xy_dataset_local"])
    assert max(after["neighbor_frame_offsets"][after["neighbor_mask"]]) <= 0
    assert not any("future" in key or "remaining" in key or "label" in key for key in before)


def test_baselines_are_causal_and_constant_velocity_is_exact():
    t = np.arange(8) * 10
    xy = np.column_stack([t * 0.3, t * -0.2])
    future = np.arange(1, 13) * 10
    b = causal_baselines(xy, t, future)
    expected = xy[-1] + future[:, None] * np.array([0.3, -0.2])
    np.testing.assert_allclose(b[1], expected)
    np.testing.assert_allclose(b[-1], expected)
    assert b.shape == (len(BASELINES), 12, 2)


def test_cache_drift_is_rejected(tmp_path):
    store(tmp_path)
    p = np.load(tmp_path / "points.npy")
    p[-1, 2] += 1
    np.save(tmp_path / "points.npy", p)
    with pytest.raises(ValueError, match="identity mismatch"):
        RecordingWindows(tmp_path)


def test_input_schema_and_label_reader_are_separate(tmp_path):
    ds = store(tmp_path)
    inputs = ds.get_inputs(0)
    assert len(inputs["causal_features"]) == len(FEATURE_NAMES)
    assert inputs["history_xy"].shape == (8, 2)
    assert inputs["neighbor_xy"].shape == (8, 8, 2)
    assert inputs["baseline_rollouts"].shape == (len(BASELINES), 12, 2)
    assert ds.identity(0)["data_role"] == "diagnostic_only"
    assert set(ds.get_labels(0)) == {"future_xy_dataset_local", "future_frame_ids"}


def test_physical_scene_cannot_cross_splits():
    catalog = [{"id": "zara1", "physical_scene": "zara", "enabled": True},
               {"id": "zara2", "physical_scene": "zara", "enabled": True}]
    with pytest.raises(ValueError, match="scene crosses"):
        validate_split_groups(catalog, {"zara1": "train", "zara2": "test"})
    validate_split_groups(catalog, {"zara1": "train", "zara2": "train"})


def test_catalog_rejects_unknown_source_and_split_aliases(tmp_path):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("1 1 2 3\n")
    audit = {"sources": [{"source": name, "raw_sha256": sha256(tmp_path / name)} for name in ("a.txt", "b.txt")]}
    config = {"source_root": ".", "recordings": [
        {"id": "same", "canonical": "a.txt", "aliases": ["b.txt"], "alias_evidence": "byte_identical"},
    ]}
    assert len(validate_catalog(config, tmp_path, audit)) == 1
    bad = deepcopy(config)
    bad["recordings"][0]["aliases"] = []
    with pytest.raises(ValueError, match="account for every"):
        validate_catalog(bad, tmp_path, audit)
    bad["recordings"].append({"id": "other", "canonical": "b.txt", "aliases": [], "alias_evidence": "single_source"})
    with pytest.raises(ValueError, match="different groups"):
        validate_catalog(bad, tmp_path, audit)
    (tmp_path / "a.txt").write_text("1 1 20 30\n")
    with pytest.raises(ValueError, match="changed source"):
        validate_catalog(config, tmp_path, audit)


def test_scene_targets_do_not_require_future_survival(tmp_path):
    p = points()
    p = p[~((p[:, 1] == 2) & (p[:, 0] > 70))]
    write_recording(tmp_path, p, {"id": "fixture", "physical_scene": "fixture_scene"})
    ds = RecordingWindows(tmp_path)
    scene = ds.get_scene_inputs(frame_id=70, horizon_raw=50)
    assert [a["agent_id"] for a in scene["agents"]] == [1, 2, 3]
    assert not scene["excluded_past_support"]
    labels = ds.get_scene_labels(scene)
    assert labels[0]["future_label_mask"].all()
    assert not labels[1]["future_label_mask"].any()
    assert labels[2]["future_label_mask"].all()
    assert not any("future" in k for agent in scene["agents"] for k in agent["inputs"])


def test_scene_membership_and_features_ignore_hidden_future(tmp_path):
    ds = store(tmp_path)
    before = ds.get_scene_inputs(frame_id=70, horizon_raw=50)
    changed = np.asarray(ds.points).copy()
    changed[changed[:, 0] > 70, 2:] = np.nan
    ds.points = changed
    after = ds.get_scene_inputs(frame_id=70, horizon_raw=50)
    assert [a["agent_id"] for a in before["agents"]] == [a["agent_id"] for a in after["agents"]]
    for a, b in zip(before["agents"], after["agents"]):
        for name in a["inputs"]:
            np.testing.assert_array_equal(a["inputs"][name], b["inputs"][name])
    assert all(not row["future_label_mask"].any() for row in ds.get_scene_labels(after))
    early = ds.get_scene_inputs(frame_id=40, horizon_raw=50)
    assert not early["agents"] and len(early["excluded_past_support"]) == 3
