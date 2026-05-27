from __future__ import annotations

from src.stage42_context_switchability_materiality_audit import _gate, _summary


def _dc() -> dict:
    return {
        "stage42_dc_gate": {"passed": 15, "total": 15},
        "baseline_family_control": {
            "protected_metric": {
                "all_improvement": 0.2877,
                "t50_improvement": 0.3154,
                "hard_failure_improvement": 0.2758,
                "easy_degradation": -0.324,
            }
        },
        "candidate_results": {
            "baseline_plus_knn_graph": {
                "test_metric": {
                    "all_improvement": 0.2881,
                    "t50_improvement": 0.3153,
                    "hard_failure_improvement": 0.2762,
                    "easy_degradation": -0.3265,
                    "switch_rate": 0.034,
                }
            },
            "baseline_plus_goal_scene": {
                "test_metric": {
                    "all_improvement": 0.2800,
                    "t50_improvement": 0.3000,
                    "hard_failure_improvement": 0.2700,
                    "easy_degradation": -0.3000,
                    "switch_rate": 0.050,
                }
            },
            "baseline_plus_scalar_neighbor": {
                "test_metric": {
                    "all_improvement": 0.2810,
                    "t50_improvement": 0.3010,
                    "hard_failure_improvement": 0.2710,
                    "easy_degradation": -0.3010,
                    "switch_rate": 0.052,
                }
            },
            "baseline_plus_graph_goal": {
                "test_metric": {
                    "all_improvement": 0.2820,
                    "t50_improvement": 0.3020,
                    "hard_failure_improvement": 0.2720,
                    "easy_degradation": -0.3020,
                    "switch_rate": 0.054,
                }
            },
            "baseline_plus_graph_history_scalar": {
                "test_metric": {
                    "all_improvement": 0.2830,
                    "t50_improvement": 0.3030,
                    "hard_failure_improvement": 0.2730,
                    "easy_degradation": -0.3030,
                    "switch_rate": 0.056,
                }
            },
        },
        "selected_context_switchability_policy": {
            "selected_candidate": "baseline_plus_knn_graph",
            "decision": "context_switchability_not_supported",
            "context_switchability_supported": False,
            "delta_vs_baseline_family_control": {
                "all_improvement": 0.0004,
                "t50_improvement": -0.0001,
                "hard_failure_improvement": 0.0004,
                "easy_degradation": -0.0024,
                "switch_rate": -0.62,
            },
        },
    }


def test_stage42_ee_summary_blocks_micro_delta_context_claim() -> None:
    summary = _summary(_dc())

    assert summary["material_context_contribution"] is False
    assert summary["decision"] == "context_switchability_materiality_blocked"
    assert summary["selected_delta_all"] < 0.01
    assert summary["selected_delta_t50"] < 0.01
    assert summary["selected_delta_hard"] < 0.01


def test_stage42_ee_gate_passes_for_negative_materiality_audit() -> None:
    payload = {
        "dc_rerun": {"stage42_dc_gate": {"passed": 15, "total": 15}},
        "summary": _summary(_dc()),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "context_main_claim_allowed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ee_context_switchability_materiality_audit_pass"
