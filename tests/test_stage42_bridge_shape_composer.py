from __future__ import annotations

from src import stage42_bridge_shape_composer as cn


def _row(name: str, all_value: float, t50: float, t100: float, hard: float, easy: float = 0.0) -> dict:
    return {
        "name": name,
        "source": "unit",
        "validation_rule": "unit",
        "status": "unit",
        "rows": 100,
        "all_improvement": all_value,
        "t50_improvement": t50,
        "t100_raw_frame_diagnostic_improvement": t100,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.2,
        "note": "unit",
    }


def test_composer_decision_blocks_missing_common_validation_switch() -> None:
    rows = [
        _row("endpoint_linear_bridge_floor", 0.21, 0.13, 0.14, 0.20),
        _row("protected_full_waypoint_sequence", 0.18, 0.15, 0.22, 0.19),
        _row("stage42j_static_gated", 0.03, 0.03, 0.02, 0.04),
    ]
    cm = {
        "deltas": {
            "full_waypoint_minus_linear_bridge": {
                "all_improvement": -0.03,
                "t50_improvement": 0.02,
                "t100_raw_frame_diagnostic_improvement": 0.08,
                "hard_failure_improvement": -0.01,
            }
        }
    }
    decision = cn._composer_decision(rows, cm)
    assert decision["full_waypoint_horizon_auxiliary_supported"] is True
    assert decision["full_waypoint_all_ade_replacement_supported"] is False
    assert decision["deployable_bridge_shape_composer_available"] is False
    assert "common validation" in decision["blocked_next_requirement"].lower()


def test_gate_passes_with_documented_blocker_and_no_overclaim() -> None:
    rows = [
        _row("endpoint_linear_bridge_floor", 0.21, 0.13, 0.14, 0.20),
        _row("protected_full_waypoint_sequence", 0.18, 0.15, 0.22, 0.19),
        _row("stage42j_static_gated", 0.03, 0.03, 0.02, 0.04),
        _row("ungated_full_waypoint_sequence", 0.30, 0.22, 0.35, 0.33, easy=1.24),
    ]
    decision = {
        "static_gated_positive_easy_safe": True,
        "full_waypoint_horizon_auxiliary_supported": True,
        "full_waypoint_all_ade_replacement_supported": False,
        "common_validation_endpoint_vs_full_waypoint_comparison_available": False,
        "blocked_next_requirement": "Build common validation-aligned endpoint-linear-vs-full-waypoint row cache.",
    }
    payload = {
        "inputs": {
            "full_waypoint": {
                "stage42_c_gate": {"passed": 12, "total": 12},
                "no_leakage": {"test_threshold_tuning": False},
            },
            "cm": {"stage42_cm_gate": {"passed": 14, "total": 14}},
            "static_gate": {
                "stage42_j_gate": {"passed": 10, "total": 10},
                "no_leakage": {"test_threshold_tuning": False},
            },
            "unified_row_cache": {
                "stage42_x_gate": {"passed": 16, "total": 16},
                "no_leakage": {"test_policy_tuning": False},
            },
        },
        "composer_decision": decision,
        "candidate_rows": rows,
        "paper_file_status": [{"contains_stage42_cn": True}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cn._gate(payload)
    assert gate["verdict"] == "stage42_cn_bridge_shape_composer_audit_pass_blocker_documented"
    assert gate["passed"] == gate["total"]
