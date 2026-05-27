from __future__ import annotations

from src.stage42_source_terms_gap_audit import _gate, _merge_rows, _summary


def _cg() -> dict:
    return {
        "stage42_cg_gate": {"passed": 11, "total": 11},
        "summary": {"terms_accepted_targets": 0, "conversion_ready_targets": 0},
        "validations": [
            {
                "dataset_id": "ucy_crowd_original",
                "conversion_ready": False,
                "terms_accepted_by_user": False,
                "cf_blockers": ["manual_terms_or_application_required"],
                "confirmation_blockers": [
                    "terms_not_accepted",
                    "terms_acceptance_date_missing",
                    "allowed_use_missing",
                    "local_path_confirmation_missing",
                    "source_identity_missing",
                ],
            },
            {
                "dataset_id": "eth_biwi_original",
                "conversion_ready": False,
                "terms_accepted_by_user": False,
                "cf_blockers": ["manual_terms_or_application_required"],
                "confirmation_blockers": [
                    "terms_not_accepted",
                    "terms_acceptance_date_missing",
                    "allowed_use_missing",
                    "local_path_confirmation_missing",
                    "source_identity_missing",
                ],
            },
        ],
    }


def _ed() -> dict:
    return {
        "stage42_ed_gate": {"passed": 15, "total": 15},
        "summary": {"technical_ready_after_terms_targets": 2},
        "action_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                "raw_path_found": True,
                "derived_cache_found": True,
                "blocker_class": "local_path_and_terms_required",
                "source_specific_metric_time_sources": ["UCY_zara01"],
                "technical_ready_source_ids_after_terms": ["UCY_zara01", "UCY_zara02", "UCY_students03"],
                "estimated_t50_windows_after_terms": 9554,
                "estimated_t100_windows_after_terms": 5605,
                "purpose": "source-specific metric/time and source-CV repair",
            },
            {
                "dataset_id": "eth_biwi_original",
                "domain": "ETH_UCY",
                "official_url": "https://vision.ee.ethz.ch/datsets.html",
                "raw_path_found": True,
                "derived_cache_found": True,
                "blocker_class": "local_path_and_terms_required",
                "source_specific_metric_time_sources": ["ETH_seq_eth"],
                "technical_ready_source_ids_after_terms": ["ETH_seq_eth", "ETH_seq_hotel"],
                "estimated_t50_windows_after_terms": 506,
                "estimated_t100_windows_after_terms": 91,
                "purpose": "source-specific metric/time and source-CV repair",
            },
            {
                "dataset_id": "trajnetplusplus_official",
                "domain": "TrajNet",
                "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
                "raw_path_found": True,
                "derived_cache_found": True,
                "blocker_class": "local_path_and_terms_required",
                "source_specific_metric_time_sources": [],
                "technical_ready_source_ids_after_terms": [],
                "estimated_t50_windows_after_terms": 0,
                "estimated_t100_windows_after_terms": 0,
                "purpose": "source-diversity acquisition or identity repair",
            },
            {
                "dataset_id": "opentraj_toolkit",
                "domain": "OpenTraj",
                "official_url": "https://github.com/crowdbotp/OpenTraj",
                "raw_path_found": True,
                "derived_cache_found": True,
                "blocker_class": "toolkit_not_independent_source",
                "source_specific_metric_time_sources": [],
                "technical_ready_source_ids_after_terms": [],
                "estimated_t50_windows_after_terms": 0,
                "estimated_t100_windows_after_terms": 0,
                "purpose": "source-diversity acquisition or identity repair",
            },
            {
                "dataset_id": "aerialmpt_or_other_topdown",
                "domain": "other_topdown",
                "official_url": "user_or_web_verified_official_url_required",
                "raw_path_found": True,
                "derived_cache_found": True,
                "blocker_class": "new_official_source_required",
                "source_specific_metric_time_sources": [],
                "technical_ready_source_ids_after_terms": [],
                "estimated_t50_windows_after_terms": 0,
                "estimated_t100_windows_after_terms": 0,
                "purpose": "source-diversity acquisition or identity repair",
            },
        ],
    }


def test_stage42_ef_prioritizes_ucy_and_missing_fields() -> None:
    rows = _merge_rows(_cg(), _ed())
    summary = _summary(rows, _cg(), _ed())

    assert rows[0]["dataset_id"] == "ucy_crowd_original"
    assert "terms_accepted_by_user" in rows[0]["missing_confirmation_fields"]
    assert "local_path" in rows[0]["missing_confirmation_fields"]
    assert summary["estimated_t50_windows_after_terms"] == 10060
    assert summary["estimated_t100_windows_after_terms"] == 5696


def test_stage42_ef_gate_passes_for_blank_terms_gap_audit() -> None:
    rows = _merge_rows(_cg(), _ed())
    payload = {
        "input_reports": {
            "stage42_cg_gate": {"passed": 11, "total": 11},
            "stage42_ed_gate": {"passed": 15, "total": 15},
        },
        "gap_rows": rows,
        "summary": _summary(rows, _cg(), _ed()),
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ef_source_terms_gap_audit_pass"
