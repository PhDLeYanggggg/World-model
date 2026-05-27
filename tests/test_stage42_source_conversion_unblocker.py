from __future__ import annotations

from src.stage42_source_conversion_unblocker import _action_rows, _gate, _summary


def _cg() -> dict:
    return {
        "stage42_cg_gate": {"passed": 11, "total": 11},
        "summary": {"terms_accepted_targets": 0},
        "validations": [
            {
                "dataset_id": "ucy_crowd_original",
                "name": "UCY",
                "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                "terms_accepted_by_user": False,
                "conversion_ready": False,
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            },
            {
                "dataset_id": "eth_biwi_original",
                "name": "ETH",
                "official_url": "https://vision.ee.ethz.ch/datsets.html",
                "terms_accepted_by_user": False,
                "conversion_ready": False,
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            },
            {
                "dataset_id": "opentraj_toolkit",
                "name": "OpenTraj",
                "official_url": "https://github.com/crowdbotp/OpenTraj",
                "terms_accepted_by_user": False,
                "conversion_ready": False,
                "confirmation_blockers": [],
                "cf_blockers": ["no_independent_t50_candidate"],
            },
            {
                "dataset_id": "aerialmpt_or_other_topdown",
                "name": "AerialMPT",
                "official_url": "user_or_web_verified_official_url_required",
                "terms_accepted_by_user": False,
                "conversion_ready": False,
                "confirmation_blockers": ["local_path_confirmation_missing"],
                "cf_blockers": ["local_path_missing"],
            },
            {
                "dataset_id": "trajnetplusplus_official",
                "name": "TrajNet++",
                "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
                "terms_accepted_by_user": False,
                "conversion_ready": False,
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            },
        ],
    }


def _dw() -> dict:
    return {
        "stage42_dw_gate": {"passed": 15, "total": 15},
        "source_rows": [
            {
                "dataset": "ucy_crowd_original",
                "source_id": "UCY_zara01",
                "technical_conversion_ready_after_terms": True,
                "horizon_counts": {"50": 1000, "100": 500},
            },
            {
                "dataset": "eth_biwi_original",
                "source_id": "ETH_seq_eth",
                "technical_conversion_ready_after_terms": True,
                "horizon_counts": {"50": 200, "100": 100},
            },
        ],
        "summary": {"domains_with_source_cv_after_terms": ["UCY"]},
    }


def _do() -> dict:
    return {
        "stage42_do_gate": {"passed": 13, "total": 13},
        "summary": {"source_specific_metric_time_sources": ["UCY_zara01", "ETH_seq_eth"]},
        "user_action_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "source_specific_metric_time_sources": ["UCY_zara01"],
                "source_specific_time_only_sources": [],
                "domain_blockers": ["source_terms_confirmation_or_conversion_readiness_missing"],
                "required_user_action": "accept terms",
            },
            {
                "dataset_id": "eth_biwi_original",
                "domain": "ETH_UCY",
                "source_specific_metric_time_sources": ["ETH_seq_eth"],
                "source_specific_time_only_sources": [],
                "domain_blockers": ["source_terms_confirmation_or_conversion_readiness_missing"],
                "required_user_action": "accept terms",
            },
        ],
    }


def _ds() -> dict:
    return {
        "stage42_ds_gate": {"passed": 13, "total": 13},
        "summary": {
            "raw_path_found_targets": 2,
            "raw_path_found_ids": ["ucy_crowd_original", "eth_biwi_original"],
            "derived_cache_found_ids": ["ucy_crowd_original"],
        },
    }


def test_stage42_ed_action_rows_preserve_legal_blockers() -> None:
    rows = _action_rows(_cg(), _dw(), _do(), _ds())
    by_id = {row["dataset_id"]: row for row in rows}

    assert by_id["ucy_crowd_original"]["technical_ready_source_count_after_terms"] == 1
    assert by_id["ucy_crowd_original"]["conversion_allowed_now"] is False
    assert by_id["ucy_crowd_original"]["estimated_t50_windows_after_terms"] == 1000
    assert by_id["opentraj_toolkit"]["blocker_class"] == "toolkit_not_independent_source"


def test_stage42_ed_gate_passes_for_unblocker_package() -> None:
    rows = _action_rows(_cg(), _dw(), _do(), _ds())
    payload = {
        "input_gates": {"cg": True, "dw": True, "do": True, "ds": True},
        "action_rows": rows,
        "summary": _summary(rows, _cg(), _dw(), _do(), _ds()),
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
    assert gate["verdict"] == "stage42_ed_source_conversion_unblocker_pass"
