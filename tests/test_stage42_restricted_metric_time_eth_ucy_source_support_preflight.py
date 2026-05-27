from pathlib import Path

from src import stage42_restricted_metric_time_eth_ucy_source_support_preflight as hk
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/restricted_metric_time_eth_ucy_source_support_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hk._build_payload()


def test_stage42_hk_augments_eth_ucy_source_support_after_terms() -> None:
    payload = _payload()
    summary = payload["summary"]
    assert summary["hj_eth_ucy_blocked_after_terms"] is True
    assert summary["eth_person_xml_candidate_sources"] >= 4
    assert summary["augmented_eth_ucy_independent_keys"] >= 5
    assert summary["augmented_eth_ucy_t100_windows_after_terms"] > 0
    assert summary["augmented_eth_ucy_source_cv_feasible_after_terms"] is True
    assert summary["augmented_eth_ucy_robust_source_cv_feasible_after_terms"] is True


def test_stage42_hk_keeps_terms_and_metric_claim_blocked() -> None:
    payload = _payload()
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    assert summary["terms_confirmed"] is False
    assert summary["conversion_ready_targets_now"] == 0
    assert claim["source_support_preflight_is_conversion"] is False
    assert claim["restricted_metric_time_claim_allowed_now"] is False
    assert claim["global_metric_claim_allowed"] is False
    assert claim["global_seconds_claim_allowed"] is False


def test_stage42_hk_gate_passes_with_terms_blocker() -> None:
    payload = _payload()
    gate = payload["stage42_hk_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hk_eth_ucy_source_support_preflight_pass_terms_blocked"


def test_stage42_hk_no_stage5c_or_smc() -> None:
    payload = _payload()
    assert payload["summary"]["conversion_executed"] is False
    assert payload["summary"]["evaluation_executed"] is False
    assert payload["summary"]["training_run"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
