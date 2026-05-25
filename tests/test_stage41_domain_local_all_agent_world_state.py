import numpy as np

from src import stage41_domain_local_all_agent_world_state as ws


def test_linear_waypoints_interpolate_endpoint() -> None:
    current = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    endpoint = np.asarray([[4.0, 0.0], [1.0, 5.0]], dtype=np.float32)
    out = ws._linear_waypoints(current, endpoint)
    assert out.shape == (2, 4, 2)
    assert np.allclose(out[0, 0], [1.0, 0.0])
    assert np.allclose(out[0, -1], [4.0, 0.0])
    assert np.allclose(out[1, 1], [1.0, 3.0])


def test_group_keys_include_source_scene_frame_and_horizon(monkeypatch) -> None:
    data = {
        "source_file": np.asarray(["a.txt", "a.txt", "b.txt"]),
        "scene_id": np.asarray(["s", "s", "s"]),
        "frame_id": np.asarray([10.2, 10.4, 10.2], dtype=np.float32),
        "agent_id": np.asarray([1, 2, 1]),
        "horizon": np.asarray([50, 50, 50], dtype=np.int16),
    }
    keys = ws._group_keys(data)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert ws._group_count(keys).tolist() == [2, 2, 1]


def test_world_state_from_selection_reports_no_future_inputs() -> None:
    data = {
        "current_xy": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "future_xy": np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        "normalizer": np.asarray([2.0, 2.0], dtype=np.float32),
        "cand_delta": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "candidate_fde": np.asarray([[0.0], [0.0]], dtype=np.float32),
        "horizon": np.asarray([50, 50], dtype=np.int16),
        "hard": np.asarray([True, False]),
        "easy": np.asarray([False, True]),
        "failure": np.asarray([False, False]),
        "domain": np.asarray(["unit", "unit"]),
        "scene_id": np.asarray(["s", "s"]),
        "source_file": np.asarray(["u", "u"]),
        "frame_id": np.asarray([1, 1], dtype=np.float32),
        "agent_id": np.asarray([1, 2]),
    }
    selected_delta = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = ws._world_state_from_selection(data, selected_delta, np.asarray([False, False]))
    assert np.allclose(out["selected_ade"], [0.0, 0.0])
    assert out["multi"].tolist() == [True, True]
    assert out["collision_delta_005"] == 0.0
