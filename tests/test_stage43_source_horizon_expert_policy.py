from src.stage43_source_horizon_expert_policy import build_source_horizon_expert_policy


def test_stage43_source_horizon_expert_policy_gate_runs_small() -> None:
    payload = build_source_horizon_expert_policy(max_trials=12, bootstrap=5)
    gate = payload["stage43_aw_gate"]
    assert gate["passed"] <= gate["total"]
    assert payload["policy"]["test_tuned"] is False
    assert payload["test_metrics"]["rows"] > 0


def test_stage43_source_horizon_expert_policy_uses_validation_selection() -> None:
    payload = build_source_horizon_expert_policy(max_trials=12, bootstrap=5)
    assert payload["eligible_validation_candidate_count"] >= 0
    assert payload["selected_validation_metrics"]["rows"] > 0
    assert payload["policy"]["selection_rule"].startswith("validation")


def test_stage43_source_horizon_expert_policy_claim_boundary() -> None:
    payload = build_source_horizon_expert_policy(max_trials=12, bootstrap=5)
    claim = payload["claim_boundary"]
    assert claim["true_3d_world_model"] is False
    assert claim["foundation_world_model"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False
    assert claim["test_threshold_tuning"] is False


def test_stage43_source_horizon_expert_policy_reports_delta() -> None:
    payload = build_source_horizon_expert_policy(max_trials=12, bootstrap=5)
    assert {"all", "t50", "hard_failure", "easy_degradation"}.issubset(payload["delta_vs_stage43_k"])
    assert "domain_easy_safe" in payload["test_flags"]
    assert "positive_t50_domain_count" in payload["test_flags"]
