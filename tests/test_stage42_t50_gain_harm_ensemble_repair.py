import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src.stage42_t50_gain_harm_ensemble_repair import _average_dicts, _gate, _row_error_bundle


def test_average_dicts_averages_matching_arrays():
    out = _average_dicts(
        [
            {"a": np.array([1.0, 3.0], dtype=np.float32), "b": np.array([2.0], dtype=np.float32)},
            {"a": np.array([3.0, 5.0], dtype=np.float32), "b": np.array([4.0], dtype=np.float32)},
        ]
    )
    assert out["a"].tolist() == [2.0, 4.0]
    assert out["b"].tolist() == [3.0]


def test_row_error_bundle_selects_neural_only_where_switched():
    frac = ft.WAYPOINT_FRAC.astype(np.float32)
    floor_waypoints = frac[:, None] * np.array([[1.0, 0.0]], dtype=np.float32)
    target_waypoints = np.stack([floor_waypoints, 2.0 * floor_waypoints], axis=0)
    labels = {
        "current_xy": np.array([[0.0, 0.0], [0.0, 0.0]]),
        "cand_delta": np.array([[[1.0, 0.0]], [[1.0, 0.0]]]),
        "normalizer": np.array([1.0, 1.0]),
        "waypoint_xy": target_waypoints,
        "waypoint_valid": np.ones((2, len(frac)), dtype=bool),
    }
    pred = {"waypoint_delta": target_waypoints.astype(np.float32)}
    switch = np.array([False, True])
    arrays = _row_error_bundle(pred, labels, switch)
    assert arrays["selected_ade"][0] == 0.0
    assert arrays["selected_ade"][1] == 0.0
    assert arrays["floor_ade"][1] > 0.0


def test_gate_passes_stable_ensemble_payload():
    payload = {
        "selector_seeds": [1, 2, 3, 4, 5, 6],
        "base_seeds": [11, 13, 17, 11, 13, 17],
        "source_labels": {"validation_policy_selection": "fresh_run"},
        "summary": {
            "ade_all": 0.05,
            "ade_t50": 0.03,
            "ade_t50_ci_low": 0.01,
            "fde_t50_ci_low": 0.02,
            "ade_hard_failure": 0.04,
            "ade_easy_degradation": 0.01,
            "trajnet_t50": 0.001,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_ii_ensemble_repair_stabilizes_t50"
    assert gate["passed"] == gate["total"]


def test_gate_flags_trajnet_blocker():
    payload = {
        "selector_seeds": [1, 2, 3, 4, 5, 6],
        "base_seeds": [11, 13, 17, 11, 13, 17],
        "source_labels": {"validation_policy_selection": "fresh_run"},
        "summary": {
            "ade_all": 0.05,
            "ade_t50": 0.03,
            "ade_t50_ci_low": 0.01,
            "fde_t50_ci_low": 0.02,
            "ade_hard_failure": 0.04,
            "ade_easy_degradation": 0.01,
            "trajnet_t50": -0.01,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["gates"]["trajnet_t50_nonnegative"] is False
    assert gate["verdict"] == "stage42_ii_ensemble_repair_partial_trajnet_blocker"
