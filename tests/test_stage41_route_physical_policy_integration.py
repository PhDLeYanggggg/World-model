import numpy as np

from src import stage41_route_physical_policy_integration as rp
from src import stage41_goal_route_physical_repair as gr


def test_route_condition_hard_route_requires_confidence():
    feat = {
        "route_idx": np.asarray([gr.ROUTE_NAMES.index("left_turn"), gr.ROUTE_NAMES.index("straight")]),
        "route_conf": np.asarray([0.9, 0.9]),
        "non_straight": np.asarray([True, False]),
        "hard_route": np.asarray([True, False]),
        "physical_challenge": np.asarray([0.8, 0.2]),
    }
    out = rp._route_condition(feat, {"route_mode": "hard_route", "route_conf_min": 0.5})
    assert out.tolist() == [True, False]


def test_improvement_delta_tracks_easy_direction():
    a = {"all_improvement": 0.2, "t50_improvement": 0.3, "hard_failure_improvement": 0.4, "easy_degradation": 0.01}
    b = {"all_improvement": 0.1, "t50_improvement": 0.4, "hard_failure_improvement": 0.2, "easy_degradation": 0.03}
    out = rp._improvement_delta(a, b)
    assert out["all_delta"] > 0
    assert out["t50_delta"] < 0
    assert out["easy_delta"] < 0


def test_slice_mapping_only_slices_row_arrays():
    mapping = {"a": np.arange(3), "b": np.arange(6).reshape(3, 2), "meta": np.arange(2)}
    out = rp._slice_mapping(mapping, np.asarray([True, False, True]))
    assert out["a"].tolist() == [0, 2]
    assert out["b"].shape == (2, 2)
    assert out["meta"].tolist() == [0, 1]
