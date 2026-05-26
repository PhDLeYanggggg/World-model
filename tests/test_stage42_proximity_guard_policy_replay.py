from __future__ import annotations

from src import stage42_proximity_guard_policy_replay as ct


def test_dict_close_reports_match_and_mismatch() -> None:
    rows = ct._dict_close({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 3.0}, ["a", "b"])
    assert rows["a"]["match"] is True
    assert rows["b"]["match"] is False


def test_policy_matches_cq_metrics_and_safety() -> None:
    policy = {
        "selected_policy": {"type": "x", "base_choices": {"A|50": True}},
        "base_choices": {"A|50": True},
        "test_summary_vs_endpoint_linear_ade": {
            "all_improvement": 0.1,
            "t50_improvement": 0.2,
            "t100_raw_frame_diagnostic_improvement": 0.3,
            "hard_failure_improvement": 0.4,
            "easy_degradation": 0.01,
            "switch_rate": 0.2,
        },
        "joint_safety_vs_endpoint_linear": {
            "near_collision_002_delta": -0.01,
            "near_collision_005_delta": -0.02,
            "p05_min_group_distance_delta": 0.01,
            "jagged_rate_delta": 0.0,
        },
    }
    cq = {
        "selected_policy": {"type": "x", "base_choices": {"A|50": True}},
        "test_eval": {"metric_vs_endpoint_ade": policy["test_summary_vs_endpoint_linear_ade"]},
        "test_joint_safety": {
            "composer_minus_endpoint": {
                "near_collision_rate_002_delta": -0.01,
                "near_collision_rate_005_delta": -0.02,
                "p05_min_group_distance_delta": 0.01,
                "jagged_rate_delta": 0.0,
            }
        },
    }
    replay = ct._policy_matches_cq(policy, cq)
    assert replay["selected_policy_match"] is True
    assert replay["base_choices_match"] is True
    assert all(row["match"] for row in replay["metric_matches"].values())
    assert all(row["match"] for row in replay["safety_matches"].values())


def _passing_payload() -> dict:
    policy = {
        "test_summary_vs_endpoint_linear_ade": {
            "all_improvement": 0.02,
            "t50_improvement": 0.01,
            "t100_raw_frame_diagnostic_improvement": 0.03,
            "hard_failure_improvement": 0.02,
            "easy_degradation": 0.001,
        },
        "joint_safety_vs_endpoint_linear": {"near_collision_005_delta": -0.001},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    return {
        "policy_artifact": {"exists": True},
        "policy_artifact_payload": policy,
        "inputs": {
            "stage42_cq": {"stage42_cq_gate": {"passed": 19, "total": 19}},
            "stage42_cr": {"stage42_cr_gate": {"passed": 19, "total": 19}},
            "stage42_cs": {"stage42_cs_gate": {"passed": 25, "total": 25}},
        },
        "replay_checks": {
            "policy_hash_recomputed_matches_cs": True,
            "policy_json_matches_cs_embedded_policy": True,
            "cq_replay": {
                "selected_policy_match": True,
                "base_choices_match": True,
                "metric_matches": {
                    "all_improvement": {"match": True},
                    "t50_improvement": {"match": True},
                    "t100_raw_frame_diagnostic_improvement": {"match": True},
                    "hard_failure_improvement": {"match": True},
                    "easy_degradation": {"match": True},
                },
                "safety_matches": {
                    "near_collision_005_delta": {"match": True},
                    "jagged_rate_delta": {"match": True},
                },
            },
            "cr_safety_policy_matches_artifact": True,
        },
    }


def test_gate_passes_for_exact_replay() -> None:
    gate = ct._gate(_passing_payload())
    assert gate["verdict"] == "stage42_ct_frozen_policy_replay_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_policy_hash_does_not_replay() -> None:
    payload = _passing_payload()
    payload["replay_checks"]["policy_hash_recomputed_matches_cs"] = False
    gate = ct._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["policy_hash_recomputed_matches_cs"] is False
