from __future__ import annotations

import numpy as np

from src.stage42_proximity_guard_batch_replay import _decision_arrays, _gate, _metric_diff
from src.stage42_proximity_guard_runtime_policy import FrozenProximityGuardPolicy


def test_decision_arrays_apply_runtime_policy_per_row() -> None:
    policy = FrozenProximityGuardPolicy(
        {
            "guard_rule": {"min_sep": 0.2, "margin": 0.005},
            "base_choices": {"ETH_UCY|50": True, "TrajNet|50": False},
        }
    )
    labels = {
        "domain": np.array(["ETH_UCY", "ETH_UCY", "TrajNet"]),
        "horizon": np.array([50, 50, 50]),
    }
    endpoint_min = np.array([0.4, 0.4, 0.4])
    candidate_min = np.array([0.3, 0.1, 0.1])
    out = _decision_arrays(policy, labels, endpoint_min, candidate_min)
    assert out["use_full"].tolist() == [True, False, False]
    assert out["guarded_off"].tolist() == [False, True, False]
    assert out["reason_counts"]["base_choice_full_waypoint_guard_clear"] == 1
    assert out["reason_counts"]["proximity_guard_fallback_to_endpoint_linear"] == 1
    assert out["reason_counts"]["base_choice_endpoint_linear"] == 1


def test_metric_diff_is_zero_for_identical_metrics() -> None:
    metric = {
        "all_improvement": 0.1,
        "t50_improvement": 0.2,
        "t100_raw_frame_diagnostic_improvement": 0.3,
        "hard_failure_improvement": 0.4,
        "easy_degradation": 0.01,
        "switch_rate": 0.2,
    }
    diff = _metric_diff(metric, dict(metric))
    assert all(value == 0.0 for value in diff.values())


def test_gate_passes_for_exact_val_test_replay() -> None:
    metric = {
        "all_improvement": 0.02,
        "t50_improvement": 0.01,
        "t100_raw_frame_diagnostic_improvement": 0.03,
        "hard_failure_improvement": 0.02,
        "easy_degradation": 0.001,
    }
    split = {
        "decision_match": True,
        "selected_xy_max_abs_diff": 0.0,
        "metric_diff_vs_expected": {
            "all_improvement": 0.0,
            "t50_improvement": 0.0,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.0,
        },
        "runtime_metric_vs_endpoint_ade": metric,
        "runtime_joint_safety": {"composer_minus_endpoint": {"near_collision_rate_005_delta": -0.001}},
    }
    payload = {
        "inputs": {
            "stage42_ct": {"stage42_ct_gate": {"passed": 30, "total": 30}},
            "stage42_cu": {"stage42_cu_gate": {"passed": 19, "total": 19}, "policy_hash": "hash"},
        },
        "policy_hash": "hash",
        "splits": {"val": split, "test": split},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_cv_batch_runtime_replay_pass"
    assert gate["passed"] == gate["total"]
