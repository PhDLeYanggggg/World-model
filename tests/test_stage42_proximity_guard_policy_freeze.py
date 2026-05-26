from __future__ import annotations

from src import stage42_proximity_guard_policy_freeze as cs


def test_policy_payload_freezes_cq_guard() -> None:
    cq = {
        "selected_policy": {
            "type": "proximity_aware_domain_horizon_full_waypoint_composer",
            "min_sep": 0.2,
            "margin": 0.005,
            "base_choices": {"ETH_UCY|50": True},
        },
        "test_eval": {
            "metric_vs_endpoint_ade": {
                "rows": 100,
                "all_improvement": 0.02,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.03,
                "hard_failure_improvement": 0.02,
                "easy_degradation": 0.001,
                "switch_rate": 0.1,
            }
        },
        "test_joint_safety": {
            "composer_minus_endpoint": {
                "near_collision_rate_002_delta": -0.001,
                "near_collision_rate_005_delta": -0.002,
                "p05_min_group_distance_delta": 0.001,
                "jagged_rate_delta": 0.0,
            }
        },
        "bootstrap_vs_endpoint_ade": {"all": {"low": 0.01}, "t50": {"low": 0.01}, "t100": {"low": 0.01}, "hard_failure": {"low": 0.01}},
        "no_leakage": {},
        "claim_boundary": {},
    }
    cr = {"deployment_recommendation": {"accuracy_priority_policy": "no_proximity_guard"}}
    policy = cs._policy_payload(cq, cr)
    assert policy["selection_scope"] == "validation_only"
    assert policy["test_usage"] == "test_once_after_policy_freeze"
    assert policy["guard_rule"]["uses_future_labels"] is False
    assert policy["accuracy_priority_diagnostic_policy"] == "no_proximity_guard"


def test_gate_requires_hash_positive_metrics_safety_and_boundaries() -> None:
    payload = {
        "inputs": {
            "stage42_cq": {"stage42_cq_gate": {"passed": 19, "total": 19}},
            "stage42_cr": {"stage42_cr_gate": {"passed": 19, "total": 19}},
        },
        "policy_artifact": {"sha256": "abc"},
        "policy_hash": "abc",
        "frozen_policy": {
            "selection_scope": "validation_only",
            "test_usage": "test_once_after_policy_freeze",
            "guard_rule": {"guard_input": "predicted endpoint/full-waypoint rollout geometry only"},
            "test_summary_vs_endpoint_linear_ade": {
                "all_improvement": 0.02,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.03,
                "hard_failure_improvement": 0.02,
                "easy_degradation": 0.001,
            },
            "joint_safety_vs_endpoint_linear": {"near_collision_005_delta": -0.001},
            "bootstrap_vs_endpoint_ade": {
                "all": {"low": 0.01},
                "t50": {"low": 0.01},
                "t100": {"low": 0.01},
                "hard_failure": {"low": 0.01},
            },
            "no_leakage": {
                "future_endpoint_input": False,
                "future_waypoints_input": False,
                "central_velocity": False,
                "test_endpoint_goals": False,
                "test_threshold_tuning": False,
            },
            "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        },
    }
    gate = cs._gate(payload)
    assert gate["verdict"] == "stage42_cs_frozen_proximity_guard_policy_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_metric_seconds_overclaim_enabled() -> None:
    payload = {
        "inputs": {
            "stage42_cq": {"stage42_cq_gate": {"passed": 19, "total": 19}},
            "stage42_cr": {"stage42_cr_gate": {"passed": 19, "total": 19}},
        },
        "policy_artifact": {"sha256": "abc"},
        "policy_hash": "abc",
        "frozen_policy": {
            "selection_scope": "validation_only",
            "test_usage": "test_once_after_policy_freeze",
            "guard_rule": {"guard_input": "predicted endpoint/full-waypoint rollout geometry only"},
            "test_summary_vs_endpoint_linear_ade": {
                "all_improvement": 0.02,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.03,
                "hard_failure_improvement": 0.02,
                "easy_degradation": 0.001,
            },
            "joint_safety_vs_endpoint_linear": {"near_collision_005_delta": -0.001},
            "bootstrap_vs_endpoint_ade": {
                "all": {"low": 0.01},
                "t50": {"low": 0.01},
                "t100": {"low": 0.01},
                "hard_failure": {"low": 0.01},
            },
            "no_leakage": {
                "future_endpoint_input": False,
                "future_waypoints_input": False,
                "central_velocity": False,
                "test_endpoint_goals": False,
                "test_threshold_tuning": False,
            },
            "claim_boundary": {"metric_or_seconds_claim": True, "stage5c_executed": False, "smc_enabled": False},
        },
    }
    gate = cs._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["metric_seconds_overclaim_blocked"] is False
