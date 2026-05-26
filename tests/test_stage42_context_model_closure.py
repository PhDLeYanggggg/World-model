from __future__ import annotations

from src import stage42_context_model_closure as dp


def _payload() -> dict:
    ar = {
        "source": "fresh_run",
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "source_overlap_pass": True,
        },
        "baseline_family_only": {"protected": {"rows": 10, "all_improvement": 0.2}},
        "sequence_variants": {
            "a": {"protected": {"all_improvement": 0.1, "t50_improvement": 0.0, "hard_failure_improvement": 0.1, "easy_degradation": 0.0}},
            "b": {"protected": {"all_improvement": 0.0, "t50_improvement": -0.1, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}},
            "c": {"protected": {"all_improvement": 0.1, "t50_improvement": -0.2, "hard_failure_improvement": -0.1, "easy_degradation": 0.0}},
        },
        "sequence_deltas": {
            "a": {"delta_vs_baseline_family_only": {"all_improvement": -0.1, "t50_improvement": -0.2, "hard_failure_improvement": -0.1, "easy_degradation": 0.0}},
            "b": {"delta_vs_baseline_family_only": {"all_improvement": -0.2, "t50_improvement": -0.3, "hard_failure_improvement": -0.2, "easy_degradation": 0.0}},
            "c": {"delta_vs_baseline_family_only": {"all_improvement": -0.1, "t50_improvement": -0.4, "hard_failure_improvement": -0.3, "easy_degradation": 0.0}},
        },
    }
    as_ = {
        "source": "fresh_run",
        "no_leakage": ar["no_leakage"],
        "baseline_family_only": {"protected": {"rows": 10, "all_improvement": 0.2}},
        "graph_variants": {
            "g1": {"protected": {"all_improvement": 0.1, "t50_improvement": 0.0, "hard_failure_improvement": 0.1, "easy_degradation": 0.0}},
            "g2": {"protected": {"all_improvement": 0.0, "t50_improvement": -0.1, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}},
            "g3": {"protected": {"all_improvement": 0.1, "t50_improvement": -0.2, "hard_failure_improvement": -0.1, "easy_degradation": 0.0}},
        },
        "graph_deltas": {
            "g1": {"delta_vs_baseline_family_only": {"all_improvement": -0.1, "t50_improvement": -0.2, "hard_failure_improvement": -0.1, "easy_degradation": 0.0}},
            "g2": {"delta_vs_baseline_family_only": {"all_improvement": -0.2, "t50_improvement": -0.3, "hard_failure_improvement": -0.2, "easy_degradation": 0.0}},
            "g3": {"delta_vs_baseline_family_only": {"all_improvement": -0.1, "t50_improvement": -0.4, "hard_failure_improvement": -0.3, "easy_degradation": 0.0}},
        },
    }
    summary = dp._summary(ar, as_, {}, {})
    return {
        "source": "fresh_synthesis_after_fresh_ar_as_rerun",
        "inputs": {
            "stage42_ar_sequence_context": {"source": "fresh_run"},
            "stage42_as_graph_context": {"source": "fresh_run"},
        },
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_summary_closes_negative_sequence_graph_protocol() -> None:
    payload = _payload()
    summary = payload["summary"]
    assert summary["closure_decision"] == "close_current_sequence_graph_residual_context_protocol"
    assert summary["positive_context_rows"] == []
    assert summary["best_delta_t50"] < 0.0


def test_gate_passes_for_fresh_negative_closure() -> None:
    payload = _payload()
    gate = dp._gate(payload)
    assert gate["verdict"] == "stage42_dp_context_model_closure_pass"
    assert gate["passed"] == gate["total"]
