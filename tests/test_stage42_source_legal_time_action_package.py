from __future__ import annotations

from src import stage42_source_legal_time_action_package as do


def test_build_action_rows_records_blocked_terms_and_calibration() -> None:
    terms = {
        "validations": [
            {
                "dataset_id": "eth_biwi_original",
                "official_url": "https://example.org/eth",
                "terms_accepted_by_user": False,
                "conversion_ready": False,
                "conversion_allowed_now": False,
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            }
        ]
    }
    time_payload = {
        "source_records": [
            {
                "domain": "ETH_UCY",
                "source_id": "ETH_seq_eth",
                "source_specific_metric_time_evidence": True,
            }
        ]
    }
    closure = {"domain_status": [{"domain": "ETH_UCY", "claim_status": "not_closed", "blockers": ["legal_terms_blocked_targets=eth_biwi_original"]}]}
    rows = do._build_action_rows(terms, time_payload, closure)
    assert rows[0]["conversion_ready"] is False
    assert rows[0]["source_specific_metric_time_sources"] == ["ETH_seq_eth"]
    assert "official terms" in rows[0]["required_user_action"]


def test_gate_passes_for_honest_blocker_package() -> None:
    payload = {
        "input_summaries": {
            "terms": {"stage42_cg_gate": {"passed": 11, "total": 11}},
            "time_geometry": {"stage42_bn_gate": {"passed": 13, "total": 13}},
            "closure": {"stage42_dd_gate": {"passed": 15, "total": 15}},
        },
        "summary": {
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "source_specific_metric_time_sources_count": 3,
        },
        "user_action_rows": [{"dataset_id": "x", "official_url": "https://example.org"} for _ in range(5)],
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "global_t100_deployable_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = do._gate(payload)
    assert gate["verdict"] == "stage42_do_source_legal_time_action_package_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_metric_overclaim() -> None:
    payload = {
        "input_summaries": {
            "terms": {"stage42_cg_gate": {"passed": 11, "total": 11}},
            "time_geometry": {"stage42_bn_gate": {"passed": 13, "total": 13}},
            "closure": {"stage42_dd_gate": {"passed": 15, "total": 15}},
        },
        "summary": {
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "source_specific_metric_time_sources_count": 1,
        },
        "user_action_rows": [{"dataset_id": "x", "official_url": "https://example.org"} for _ in range(5)],
        "claim_boundary": {
            "global_metric_claim_allowed": True,
            "global_seconds_claim_allowed": False,
            "global_t100_deployable_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = do._gate(payload)
    assert gate["gates"]["global_metric_seconds_blocked"] is False
    assert gate["passed"] == gate["total"] - 1
