from src.stage43_source_horizon_safety_envelope import build_source_horizon_safety_envelope


def test_stage43_source_horizon_safety_envelope_gate_passes_small() -> None:
    payload = build_source_horizon_safety_envelope(max_trials=4)
    gate = payload["stage43_av_gate"]
    assert gate["verdict"] == "stage43_av_source_horizon_safety_envelope_pass"
    assert gate["passed"] == gate["total"]


def test_stage43_source_horizon_safety_envelope_is_diagnostic_not_deployment() -> None:
    payload = build_source_horizon_safety_envelope(max_trials=4)
    assert payload["test_diagnostic_only"] is True
    assert payload["deployment_selection_from_test"] is False
    assert payload["claim_boundary"]["test_selection_for_deployment"] is False
    assert payload["stage43_av_gate"]["deploy_new_policy"] is False


def test_stage43_source_horizon_safety_envelope_reports_counts() -> None:
    payload = build_source_horizon_safety_envelope(max_trials=4)
    summary = payload["envelope_summary"]
    assert payload["trial_count"] == 4
    assert summary["aggregate_safe_trial_count"] >= 0
    assert summary["domain_easy_safe_trial_count"] >= 0
    assert summary["deployable_like_trial_count"] >= 0
    assert summary["best_test_t50_trial"]["name"]


def test_stage43_source_horizon_safety_envelope_claim_boundary() -> None:
    payload = build_source_horizon_safety_envelope(max_trials=4)
    claim = payload["claim_boundary"]
    assert claim["true_3d_world_model"] is False
    assert claim["foundation_world_model"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False
