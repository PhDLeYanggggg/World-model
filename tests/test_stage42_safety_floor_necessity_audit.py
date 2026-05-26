from src import stage42_safety_floor_necessity_audit as bw


def test_gate_passes_when_floor_needed_and_claims_are_bounded() -> None:
    payload = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "input_gates": {
            "a": {"passed": 1, "total": 1},
            "b": {"passed": 2, "total": 2},
        },
        "summary": {
            "current_all_improvement": 0.1,
            "current_t50_improvement": 0.1,
            "current_hard_failure_improvement": 0.1,
            "current_easy_degradation": 0.0,
            "floor_free_neural_deployable": False,
            "baseline_family_rollout_context_supported": True,
            "small_tabular_neural_context_supported": False,
            "source_blockers_active": 1,
        },
        "safety_floor_findings": {
            "ungated_endpoint_easy_violation": True,
            "ungated_full_waypoint_easy_violation": True,
        },
        "context_findings": {
            "fallback_removal_for_baseline_family_probe": "supported_on_this_source_level_split",
            "teacher_floor_context_removal": "not_supported_as_global_replacement",
            "context_removal_hurts_protected_t50": True,
        },
        "mechanism_findings": {
            "dominant_supported_mechanism": "baseline_family_rollout_context_supported_as_dominant_mechanism",
            "neural_context_not_supported": True,
        },
        "row_level_findings": {
            "row_level_positive_and_easy_safe": True,
            "unified_cache_positive_and_easy_safe": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bw._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bw_safety_floor_necessity_audit_pass"


def test_ci_low_helper_returns_zero_for_missing_rows() -> None:
    assert bw._ci_low({}, "missing") == 0.0
    assert bw._ci_low({"metric": {"ci_low": 0.123}}, "metric") == 0.123


def test_context_delta_reads_nested_stage42_at_schema() -> None:
    row = {"protected_delta_vs_all_context": {"t50_improvement": -0.092}}
    assert bw._context_delta(row, "t50_improvement") == -0.092
    assert bw._context_delta({"t50_improvement": -0.01}, "t50_improvement") == -0.01
