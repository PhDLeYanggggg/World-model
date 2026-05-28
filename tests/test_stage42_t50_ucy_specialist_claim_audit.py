from src import stage42_t50_ucy_specialist_claim_audit as il


def test_gate_passes_guarded_claim_audit_payload():
    payload = {
        "summary": {
            "stage42ii_verdict": "stage42_ii_ensemble_repair_stabilizes_t50",
            "stage42ij_verdict": "stage42_ij_t50_ensemble_source_robustness_pass",
            "stage42ik_verdict": "stage42_ik_ucy_specialist_integration_pass",
            "stage42x_verdict": "stage42_x_unified_row_level_full_waypoint_cache_pass",
            "ucy_delta": {"before_t50": 0.0, "after_t50": 0.1, "delta_t50": 0.1},
            "global_delta": {"ade_all_delta_vs_stage42ii": 0.02, "ade_t50_delta_vs_stage42ii": 0.03, "easy_degradation_delta_vs_stage42ii": 0.0},
            "non_ucy_max_abs_delta": 0.0,
            "all_powered_t50_sources_positive": True,
            "supported_claim_count": 4,
            "blocked_claim_count": 4,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "source_specialist_claim_only": True,
            "independent_new_domain_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = il._gate(payload)
    assert gate["verdict"] == "stage42_il_ucy_specialist_claim_audit_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_independent_domain_overclaim():
    payload = {
        "summary": {
            "stage42ii_verdict": "stage42_ii_ensemble_repair_stabilizes_t50",
            "stage42ij_verdict": "stage42_ij_t50_ensemble_source_robustness_pass",
            "stage42ik_verdict": "stage42_ik_ucy_specialist_integration_pass",
            "stage42x_verdict": "stage42_x_unified_row_level_full_waypoint_cache_pass",
            "ucy_delta": {"before_t50": 0.0, "after_t50": 0.1, "delta_t50": 0.1},
            "global_delta": {"ade_all_delta_vs_stage42ii": 0.02, "ade_t50_delta_vs_stage42ii": 0.03, "easy_degradation_delta_vs_stage42ii": 0.0},
            "non_ucy_max_abs_delta": 0.0,
            "all_powered_t50_sources_positive": True,
            "supported_claim_count": 4,
            "blocked_claim_count": 4,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "source_specialist_claim_only": True,
            "independent_new_domain_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = il._gate(payload)
    assert gate["gates"]["scope_not_overclaimed"] is False


def test_stage42_il_run_records_ucy_delta_and_boundaries():
    result = il.run_stage42_t50_ucy_specialist_claim_audit()
    gate = result["stage42_il_gate"]
    assert gate["gates"]["ucy_t50_repaired"]
    assert gate["gates"]["non_ucy_unchanged"]
    assert result["summary"]["ucy_delta"]["after_t50"] > 0.0
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
