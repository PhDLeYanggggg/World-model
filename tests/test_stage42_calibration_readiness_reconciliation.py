from src.stage42_calibration_readiness_reconciliation import _gate, _summary


def _payload(**summary_overrides):
    summary = {
        "required_dataset_coverage_complete": True,
        "direct_path_groups_checked": 9,
        "direct_path_groups_found": 7,
        "source_specific_candidate_count": 6,
        "restricted_terms_confirmed": False,
        "restricted_metric_time_ready_now": False,
        "restricted_conversion_ready_targets_now": 0,
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "stage42_b_external_validation_ready": True,
        "stage42_c_full_waypoint_prereq_ready": True,
    }
    summary.update(summary_overrides)
    return {
        "summary": summary,
        "input_gate_status": {
            "data_calibration_present": True,
            "source_time_geometry_gate_passed": True,
            "restricted_metric_time_claim_guard_passed": True,
            "calibration_manifest_gate_passed": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "global_metric_claim": False,
            "global_seconds_claim": False,
            "restricted_metric_time_claim_allowed_now": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_for_blocked_but_covered_calibration_state():
    gate = _gate(_payload())
    assert gate["verdict"] == "stage42_jd_calibration_readiness_reconciliation_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_restricted_metric_time_is_marked_ready():
    gate = _gate(_payload(restricted_metric_time_ready_now=True))
    assert gate["gates"]["restricted_ready_now_false"] is False
    assert gate["verdict"] == "stage42_jd_calibration_readiness_reconciliation_partial"


def test_gate_fails_on_global_metric_overclaim():
    payload = _payload()
    payload["claim_boundary"]["global_metric_claim"] = True
    gate = _gate(payload)
    assert gate["gates"]["global_metric_blocked"] is False


def test_summary_keeps_ready_now_false_even_with_candidate_hints():
    summary = _summary(
        {"summary": {"datasets_audited": 7, "stage42_b_external_validation_ready": True, "stage42_c_full_waypoint_prereq_ready": True}},
        {"summary": {"source_specific_metric_time_sources": ["ETH_seq_eth", "UCY_zara01"]}},
        {"summary": {"hk_terms_confirmed": False, "hk_ready_now": False, "hk_conversion_ready_targets_now": 0}},
        {"summary": {"conversion_ready_targets": 0, "converted_datasets_now": 0, "evaluated_datasets_now": 0}},
        {"groups_checked": 9, "groups_found": 7},
        [{"dataset": name} for name in ["sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"]],
        [{"source_id": "ETH_seq_eth"}, {"source_id": "UCY_zara01"}],
    )
    assert summary["restricted_metric_time_ready_now"] is False
    assert summary["restricted_metric_time_claim_allowed_now"] is False
    assert summary["required_dataset_coverage_complete"] is True
