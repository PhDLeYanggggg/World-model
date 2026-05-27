from pathlib import Path

from src import stage42_restricted_metric_time_terms_intake_v2 as hm
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hm._build_payload()


def test_stage42_hm_builds_source_level_candidates_for_ucy_and_eth_ucy() -> None:
    payload = _payload()
    summary = payload["summary"]
    assert summary["source_level_candidates"] >= 10
    assert set(summary["after_terms_domains_with_source_cv_candidate_count"]).issuperset({"UCY", "ETH_UCY"})
    assert summary["after_terms_total_t50_windows"] > 0
    assert summary["after_terms_total_t100_windows"] > 0


def test_stage42_hm_blank_template_blocks_ready_now() -> None:
    payload = _payload()
    summary = payload["summary"]
    assert summary["conversion_ready_candidates_now"] == 0
    assert summary["ready_now_t50_windows"] == 0
    assert summary["ready_now_t100_windows"] == 0
    assert payload["claim_boundary"]["restricted_metric_time_claim_allowed_now"] is False
    assert payload["claim_boundary"]["global_metric_claim_allowed"] is False
    assert payload["claim_boundary"]["global_seconds_claim_allowed"] is False


def test_stage42_hm_validator_rejects_eth_person_without_user_verified_official_terms(tmp_path: Path) -> None:
    local = tmp_path / "eth_person"
    local.mkdir()
    candidate = {
        "candidate_id": "hk::ETH-Person_bahnhof_assc_gt",
        "source_id": "ETH-Person_bahnhof_assc_gt",
        "domain": "ETH_UCY",
        "terms_target_id": "eth_person_local_candidates",
        "official_terms_url_hint": "user_verified_official_eth_person_source_terms_url_required",
        "source_cv_usable_after_terms": True,
        "t50_windows_after_terms": 10,
        "t100_windows_after_terms": 5,
    }
    template_row = {
        "user_confirmation": {
            "terms_accepted_by_user": True,
            "terms_acceptance_date": "2026-05-27",
            "official_terms_url": "https://example.edu/eth-person",
            "accepted_terms_version_or_access_date": "2026-05-27",
            "allowed_use": "research_only",
            "redistribution_allowed": "false",
            "derived_data_allowed": "true",
            "local_path": str(local),
            "source_identity": "ETH-Person_bahnhof_assc_gt",
            "confirmed_by_user": "confirmed",
        }
    }
    validation = hm.validate_candidate(candidate, template_row)
    assert validation["conversion_ready"] is False
    assert "official_terms_url_requires_user_verified_official_source" in validation["blockers"]


def test_stage42_hm_accepts_filled_ucy_candidate_for_future_guarded_conversion(tmp_path: Path) -> None:
    local = tmp_path / "ucy"
    local.mkdir()
    candidate = {
        "candidate_id": "hj::UCY_zara02",
        "source_id": "UCY_zara02",
        "domain": "UCY",
        "terms_target_id": "ucy_crowd_original",
        "official_terms_url_hint": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "source_cv_usable_after_terms": True,
        "t50_windows_after_terms": 100,
        "t100_windows_after_terms": 50,
    }
    template_row = {
        "user_confirmation": {
            "terms_accepted_by_user": True,
            "terms_acceptance_date": "2026-05-27",
            "official_terms_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
            "accepted_terms_version_or_access_date": "2026-05-27",
            "allowed_use": "research_only",
            "redistribution_allowed": "false",
            "derived_data_allowed": "true",
            "local_path": str(local),
            "source_identity": "UCY_zara02",
            "confirmed_by_user": "confirmed by user",
        }
    }
    validation = hm.validate_candidate(candidate, template_row)
    assert validation["conversion_ready"] is True
    assert validation["restricted_metric_time_ready_now"] is False
    assert validation["converted_now"] is False
    assert validation["evaluated_now"] is False


def test_stage42_hm_gate_passes_blocked_until_user_confirmation() -> None:
    payload = _payload()
    gate = payload["stage42_hm_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hm_restricted_metric_time_terms_intake_v2_pass_blocked_until_user_confirmation"
    assert payload["summary"]["download_executed"] is False
    assert payload["summary"]["conversion_executed"] is False
    assert payload["summary"]["evaluation_executed"] is False
    assert payload["summary"]["training_executed"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
