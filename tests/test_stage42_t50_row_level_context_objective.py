from src import stage42_t50_row_level_context_objective as kb


_PAYLOAD = None


def _payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = kb.run_stage42_t50_row_level_context_objective(refresh_readmes=False)
    return _PAYLOAD


def test_stage42_kb_gate_passes() -> None:
    payload = _payload()
    gate = payload["stage42_kb_gate"]
    assert gate["verdict"] == "stage42_kb_t50_row_level_context_objective_pass"
    assert gate["passed"] == gate["total"]


def test_stage42_kb_uses_validation_only_selection() -> None:
    payload = _payload()
    best = payload["summary"]["best_trial"]
    assert best["validation_selection"]["source"] == "validation_only"
    assert best["validation_selection"]["test_threshold_tuning"] is False
    assert best["rows"]["train"] > 0
    assert best["rows"]["val"] > 0
    assert best["rows"]["test"] > 1000


def test_stage42_kb_records_oracle_and_switch_diagnostics() -> None:
    payload = _payload()
    best = payload["summary"]["best_trial"]
    assert best["oracle_metric_vs_baseline_family"]["t50_improvement"] > 0
    assert "capture_rate" in best["switch_diagnostics"]
    assert "switched_harm_rate" in best["switch_diagnostics"]
    assert payload["summary"]["failure_or_success_reason"]


def test_stage42_kb_claim_boundaries() -> None:
    payload = _payload()
    claim = payload["claim_boundary"]
    assert claim["true_3d"] is False
    assert claim["foundation_world_model"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False
    assert all(payload["no_leakage"].values())
