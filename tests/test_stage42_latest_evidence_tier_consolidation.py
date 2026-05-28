from src.stage42_latest_evidence_tier_consolidation import _gate, _summary


def _payload(**overrides):
    summary = {
        "decision": "latest_evidence_tiers_consolidated_context_not_promoted",
        "main_evidence_tier": "T1_source_level_row_cache_full_waypoint",
        "main_all_improvement": 0.29,
        "main_t50_improvement": 0.24,
        "main_t100_raw_frame_diagnostic_improvement": 0.19,
        "main_hard_failure_improvement": 0.28,
        "main_easy_degradation": 0.0,
        "context_slice_local_support": 14,
        "context_policy_promotable": False,
        "ja_decision": "validation_selected_context_slice_policy_not_promoted",
        "jb_decision": "conservative_context_slice_policy_not_promoted",
    }
    summary.update(overrides.pop("summary", {}))
    rows = [
        {"tier": "T1_source_level_row_cache_full_waypoint", "status": "main_supported_evidence"},
        {"tier": "T2_mechanism_row_cache_audit", "status": "mechanism_supported"},
        {"tier": "T3_context_slice_analysis", "status": "local_slice_supported_not_deployable"},
        {"tier": "T4_context_policy_promotion", "status": "not_promotable"},
        {"tier": "T5_conservative_context_repair", "status": "not_promotable"},
        {"tier": "T6_module_claim_lock", "status": "claim_lock_passed"},
    ]
    payload = {
        "summary": summary,
        "tier_rows": rows,
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
    payload.update(overrides)
    return payload


def test_gate_passes_for_latest_tier_consolidation():
    gate = _gate(_payload())
    assert gate["verdict"] == "stage42_jc_latest_evidence_tier_consolidation_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_context_is_over_promoted():
    payload = _payload(summary={"context_policy_promotable": True})
    gate = _gate(payload)
    assert gate["gates"]["context_policy_not_promoted"] is False
    assert gate["verdict"] == "stage42_jc_latest_evidence_tier_consolidation_partial"


def test_gate_fails_on_metric_seconds_overclaim():
    payload = _payload()
    payload["claim_boundary"]["metric_or_seconds_claim"] = True
    gate = _gate(payload)
    assert gate["gates"]["t100_reported_as_raw_diagnostic"] is False
    assert gate["gates"]["no_metric_seconds_claim"] is False


def test_summary_keeps_context_local_when_promotion_fails():
    inputs = {
        "iv": {
            "metric": {
                "all_improvement": 0.29,
                "t50_improvement": 0.24,
                "t100_improvement": 0.19,
                "hard_failure_improvement": 0.28,
                "easy_degradation": 0.0,
            }
        },
        "iz": {"summary": {"supported_context_slice_count": 14}},
        "ja": {"summary": {"decision": "validation_selected_context_slice_policy_not_promoted"}},
        "jb": {"summary": {"decision": "conservative_context_slice_policy_not_promoted"}},
        "gj": {"summary": {"supported_main_modules_locked": ["history"], "blocked_main_modules_locked": ["scene_goal"]}},
    }
    summary = _summary(inputs, [{"tier": "T1_source_level_row_cache_full_waypoint", "rows": 100}])
    assert summary["context_slice_local_support"] == 14
    assert summary["context_policy_promotable"] is False
    assert "context modules remain slice-local" in summary["paper_ready_claim"]
