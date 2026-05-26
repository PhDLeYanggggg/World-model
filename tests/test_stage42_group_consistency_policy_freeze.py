from __future__ import annotations

from src import stage42_group_consistency_policy_freeze as dj


def _di_payload() -> dict:
    return {
        "repair": {
            "selected": {
                "candidate": {"mode": "repel_unsafe", "min_sep": 0.08, "margin": 0.0, "strength": 0.5},
                "val_score": 1.0,
                "val_metric": {"all_improvement": 0.2},
                "val_diagnostics": {"final_near_005": 0.01},
            },
            "test": {
                "metric_vs_floor": {
                    "rows": 10,
                    "all_improvement": 0.2,
                    "t50_improvement": 0.1,
                    "t100_raw_frame_diagnostic_improvement": 0.05,
                    "hard_failure_improvement": 0.12,
                    "easy_degradation": 0.0,
                    "switch_rate": 0.5,
                },
                "diagnostics": {
                    "base_near_005": 0.02,
                    "final_near_005": 0.01,
                    "floor_near_005": 0.03,
                    "base_p05_min_distance": 0.07,
                    "final_p05_min_distance": 0.08,
                    "floor_p05_min_distance": 0.06,
                    "unsafe_rows": 2,
                    "unsafe_rate": 0.2,
                },
                "bootstrap": {"all": {"low": 0.1}},
            },
        },
        "comparison_to_prior": {
            "delta_vs_stage42_am": {"all_improvement": 0.01, "t50_improvement": 0.02, "hard_failure_improvement": 0.01},
            "delta_vs_stage42_cq": {},
            "delta_vs_stage42_dh": {},
        },
        "group_schema": {"key": "source_file + frame_id*1000 + horizon"},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_policy_payload_records_repair_rule_and_boundaries() -> None:
    policy = dj._policy_payload(_di_payload())
    assert policy["selection_scope"] == "validation_only"
    assert policy["test_usage"] == "test_once_from_stage42_di_after_validation_selection"
    assert policy["repair_rule"]["type"] == "repel_unsafe"
    assert policy["repair_rule"]["uses_future_labels"] is False
    assert policy["test_group_safety"]["final_near_005"] < policy["test_group_safety"]["base_near_005"]


def test_gate_passes_for_positive_frozen_policy() -> None:
    policy = dj._policy_payload(_di_payload())
    payload = {
        "inputs": {"stage42_di": {"stage42_di_gate": {"passed": 17, "total": 17}}},
        "policy_artifact": {"sha256": "abc"},
        "policy_hash": "abc",
        "frozen_policy": policy,
        "paper_file_status": [{"exists": True, "contains_stage42_dj": True}],
    }
    gate = dj._gate(payload)
    assert gate["verdict"] == "stage42_dj_frozen_group_consistency_policy_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_test_threshold_tuning_is_true() -> None:
    policy = dj._policy_payload(_di_payload())
    policy["no_leakage"]["test_threshold_tuning"] = True
    payload = {
        "inputs": {"stage42_di": {"stage42_di_gate": {"passed": 17, "total": 17}}},
        "policy_artifact": {"sha256": "abc"},
        "policy_hash": "abc",
        "frozen_policy": policy,
        "paper_file_status": [{"exists": True, "contains_stage42_dj": True}],
    }
    gate = dj._gate(payload)
    assert gate["gates"]["no_test_threshold_tuning"] is False
    assert gate["passed"] == gate["total"] - 1
