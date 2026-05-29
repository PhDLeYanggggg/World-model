from __future__ import annotations

from src import stage42_current_module_claim_refresh as jt


def test_current_claim_refresh_preserves_positive_row_cache_and_blocks_incremental_overclaim() -> None:
    payload = jt.run_stage42_current_module_claim_refresh(refresh_readmes=False)
    summary = payload["summary"]

    assert summary["row_cache"]["ade_all"] > 0.0
    assert summary["row_cache"]["ade_t50"] > 0.0
    assert summary["row_cache"]["ade_hard_failure"] > 0.0
    assert summary["row_cache"]["easy_degradation"] <= 0.02
    assert "history_only" in summary["ao"]["positive_standalone_context_variants"]
    assert summary["ao"]["positive_incremental_context_variants"] == []
    assert "incremental_context_after_baseline_family" in summary["blocked_independent_claims"]


def test_current_claim_refresh_gate_passes_with_claim_boundaries() -> None:
    payload = jt.run_stage42_current_module_claim_refresh(refresh_readmes=False)
    gate = jt._gate(payload)

    assert gate["verdict"] == "stage42_jt_current_module_claim_refresh_pass"
    assert gate["passed"] == gate["total"]
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False


def test_current_claim_refresh_gate_catches_overclaim() -> None:
    payload = jt.run_stage42_current_module_claim_refresh(refresh_readmes=False)
    broken = dict(payload)
    broken["summary"] = dict(payload["summary"])
    broken["summary"]["blocked_independent_claims"] = [
        item for item in payload["summary"]["blocked_independent_claims"] if item != "incremental_context_after_baseline_family"
    ]

    gate = jt._gate(broken)

    assert gate["gates"]["incremental_context_not_overclaimed"] is False
    assert gate["verdict"] == "stage42_jt_current_module_claim_refresh_partial"
