import numpy as np
import pytest
import torch

from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter, dense_prefix
from src.data_unification.m3w_causal_recordings import RecordingWindows, write_recording
from src.world_model.m3w_offline_visual_forecast import geometry_features
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_supervised_intervention import build_forecaster


def points():
    return np.array([[t, agent, t * (agent + 1.) / 2, agent * 5.] for agent in range(3)
                     for t in range(24)], dtype=np.float64)


def adapter(p=None, query=7):
    p = points() if p is None else p
    return ExternalPrefixAdapter(p[p[:, 0] <= query], query_frame=query, recording_id="synthetic_only")


def test_complete_prefix_matches_existing_causal_input_schema(tmp_path):
    p = points()
    write_recording(tmp_path, p, {"id": "synthetic_only", "physical_scene": "synthetic"})
    original = RecordingWindows(tmp_path).get_scene_inputs(7, 12, history_steps=8)
    geometry, scene = adapter().geometry_batch()
    expected = np.stack([geometry_features(a["inputs"]) for a in original["agents"]])
    np.testing.assert_array_equal(geometry, expected)
    assert scene["observed_agent_ids"].tolist() == [0, 1, 2]
    assert not scene["source_admission"]


def test_future_changes_cannot_change_prefix_or_agent_inventory():
    p = points()
    a, scene_a = adapter(p).geometry_batch()
    p[p[:, 0] > 7, 2:] = 1e20
    b, scene_b = adapter(p).geometry_batch()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(scene_a["observed_agent_ids"], scene_b["observed_agent_ids"])
    p = p[p[:, 0] <= 7]
    c, _ = adapter(p).geometry_batch()
    np.testing.assert_array_equal(a, c)


def test_incomplete_ego_remains_observed_neighbor_without_future_filter():
    p = points()
    p = p[~((p[:, 1] == 2) & (p[:, 0] < 6))]
    geometry, scene = adapter(p).geometry_batch()
    assert geometry.shape == (2, 476)
    assert scene["observed_agent_ids"].tolist() == [0, 1, 2]
    assert scene["excluded_past_support"] == [{"agent_id": 2, "reason": "insufficient_past"}]
    first = scene["agents"][0]["inputs"]
    assert first["neighbor_mask"].sum(1).tolist()[:2] == [8, 2]
    assert (first["neighbor_frame_offsets"][first["neighbor_mask"]] <= 0).all()


def test_gap_velocity_not_invented_for_latest_neighbor():
    p = np.array([[t, 0, float(t), 0.] for t in range(8)] +
                 [[5, 1, 10., 0.], [7, 1, 8., 0.]], dtype=np.float64)
    _, scene = adapter(p).geometry_batch()
    first = scene["agents"][0]["inputs"]
    assert first["causal_features"][9] == 10.
    assert first["causal_features"][10] == 0.
    assert first["neighbor_mask"][0].sum() == 2


@pytest.mark.parametrize("change", ["future", "duplicate", "nan", "fractional", "negative", "missing_current"])
def test_malformed_or_future_prefix_refused(change):
    p = points()[points()[:, 0] <= 7].copy()
    if change == "future":
        p[0, 0] = 8
    elif change == "duplicate":
        p = np.r_[p, p[:1]]
    elif change == "nan":
        p[0, 2] = np.nan
    elif change == "fractional":
        p[0, 0] = .5
    elif change == "negative":
        p[0, 1] = -1
    else:
        p = p[p[:, 0] < 7]
    with pytest.raises(ValueError):
        ExternalPrefixAdapter(p, query_frame=7, recording_id="synthetic_only")


def test_no_labels_or_future_conditioned_item_api():
    reader = adapter()
    for method in [reader.get_labels, reader.get_scene_labels]:
        with pytest.raises(PermissionError, match="no future-label"):
            method(0)
    with pytest.raises(ValueError, match="shared-scene"):
        reader.get_inputs(0)


def test_empty_eligible_targets_keeps_observed_inventory():
    p = np.array([[7, 0, 1., 2.], [7, 1, 2., 3.]], dtype=np.float64)
    geometry, scene = adapter(p).geometry_batch()
    assert geometry.shape == (0, 476)
    assert scene["observed_agent_ids"].tolist() == [0, 1]
    assert len(scene["excluded_past_support"]) == 2


def test_real_torch_input_interface_runs_without_labels():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = build_forecaster(dict(width=8, heads=2, layers=1, neighbor_policy="complete_aligned_history",
        input_conditioning="observed_joint_max_norm", output_parameterization="motion_bounded"))
    geometry, _ = adapter().geometry_batch()
    inputs = pack_geometry(geometry)
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
    assert prediction.shape == (3, 12, 2) and torch.isfinite(prediction).all()
    assert not {"target", "future_mask", "labels", "source_id", "agent_id"} & inputs.keys()


def dense_fixture():
    positions = np.full((4, 24, 2), np.nan, dtype=np.float64)
    valid = np.zeros((4, 24), dtype=bool)
    p = points()
    for frame, agent, x, y in p:
        positions[int(agent), int(frame)] = [x, y]
        valid[int(agent), int(frame)] = True
    positions[3, 8:] = 1e6
    valid[3, 8:] = True
    return positions, valid, np.arange(4)


def test_dense_prefix_never_includes_future_only_agents_or_coordinates():
    positions, valid, ids = dense_fixture()
    p = dense_prefix(positions, valid, ids, query_frame=7)
    expected, _ = adapter().geometry_batch()
    np.testing.assert_array_equal(adapter(p).geometry_batch()[0], expected)
    positions[:, 8:] = 1e20
    valid[:, 8:] = ~valid[:, 8:]
    np.testing.assert_array_equal(dense_prefix(positions, valid, ids, query_frame=7), p)
    assert set(p[:, 1]) == {0, 1, 2}


def test_dense_source_only_reads_the_explicit_prefix_slice():
    class SliceFence:
        def __init__(self, values):
            self.values, self.shape, self.dtype = values, values.shape, values.dtype
        def __getitem__(self, key):
            assert key[1] == slice(3, 11)
            return self.values[key]
        def __array__(self, *args, **kwargs):
            raise AssertionError("Whole recording materialization is forbidden")
    positions, valid, ids = dense_fixture()
    p = dense_prefix(SliceFence(positions), SliceFence(valid), ids, query_frame=10)
    assert p[:, 0].min() == 3 and p[:, 0].max() == 10
    assert len(p) == 27


@pytest.mark.parametrize("change", ["shape", "mask_dtype", "ids", "nonfinite", "query"])
def test_dense_source_rejects_invalid_observed_prefix(change):
    positions, valid, ids = dense_fixture()
    query = 7
    if change == "shape":
        valid = valid[:, :-1]
    elif change == "mask_dtype":
        valid = valid.astype(np.int8)
    elif change == "ids":
        ids[1] = ids[0]
    elif change == "nonfinite":
        positions[0, 7] = np.nan
    else:
        query = 30
    with pytest.raises(ValueError):
        dense_prefix(positions, valid, ids, query_frame=query)


def test_dense_source_empty_current_inventory_is_not_a_future_filter():
    positions, valid, ids = dense_fixture()
    valid[:, 7] = False
    p = dense_prefix(positions, valid, ids, query_frame=7)
    assert len(p) == 21
    with pytest.raises(ValueError, match="currently observed"):
        adapter(p)
