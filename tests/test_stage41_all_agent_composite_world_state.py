import numpy as np

from src import stage41_all_agent_composite_world_state as comp


def test_group_count_matches_same_key_rows() -> None:
    keys = np.asarray(["a", "a", "b", "c", "c", "c"], dtype=object)
    assert comp._group_count(keys).tolist() == [2, 2, 1, 3, 3, 3]


def test_selected_world_state_blends_all_waypoints(monkeypatch) -> None:
    data = {
        "floor_xy": np.asarray([[[0.0, 0.0], [2.0, 0.0]]]),
        "neural_xy": np.asarray([[[2.0, 0.0], [4.0, 0.0]]]),
        "labels": {
            "waypoint_xy": np.asarray([[[1.0, 0.0], [3.0, 0.0]]]),
            "waypoint_valid": np.asarray([[True, True]]),
        },
    }

    monkeypatch.setattr(comp.blend, "_alpha_vector", lambda _data, _policy: np.asarray([0.5]))
    out = comp._selected_world_state(data, {"type": "unit"})
    assert np.allclose(out["selected_xy"], [[[1.0, 0.0], [3.0, 0.0]]])
    assert np.allclose(out["selected_ade"], [0.0])
    assert out["switch"].tolist() == [True]
