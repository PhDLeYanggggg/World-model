from __future__ import annotations

from src.stage43_source_level_caveat_audit import run_audit


def test_stage43_j_blocks_uniform_source_overclaim():
    payload = run_audit()
    assert payload["stage43_j_gate"]["verdict"] == "stage43_j_source_level_caveat_mapped"
    assert payload["source_uniform_candidate"] is False
    assert payload["domain_level_candidate"] is True
    assert payload["nonpositive_source_count"] >= 1
    assert payload["claim_boundary"]["uniform_per_source_claim"] is False


def test_stage43_j_does_not_attempt_test_tuned_repair():
    payload = run_audit()
    assert payload["repair_attempted"] is False
    forbidden = payload["recommended_next_action"]["forbidden_methods"]
    assert "choose thresholds from test source metrics" in forbidden
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
