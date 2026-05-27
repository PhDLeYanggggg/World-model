from __future__ import annotations

from src.stage42_source_terms_confirmation_intake import REQUIRED_FIELDS, _gate, _intake_rows, _schema, _summary


def _ef() -> dict:
    return {
        "stage42_ef_gate": {"passed": 13, "total": 13},
        "gap_rows": [
            {
                "dataset_id": "eth_biwi_original",
                "domain": "ETH_UCY",
                "official_url": "https://vision.ee.ethz.ch/datsets.html",
                "estimated_t50_windows_after_terms": 506,
                "estimated_t100_windows_after_terms": 91,
                "source_cv_after_terms": True,
                "technical_ready_source_ids_after_terms": ["ETH_seq_eth", "ETH_seq_hotel"],
                "blocker_class": "local_path_and_terms_required",
                "missing_confirmation_fields": ["terms_accepted_by_user", "local_path", "source_identity"],
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            },
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                "estimated_t50_windows_after_terms": 9554,
                "estimated_t100_windows_after_terms": 5605,
                "source_cv_after_terms": True,
                "technical_ready_source_ids_after_terms": ["UCY_zara01", "UCY_zara02", "UCY_students03"],
                "blocker_class": "local_path_and_terms_required",
                "missing_confirmation_fields": ["terms_accepted_by_user", "local_path", "source_identity"],
                "confirmation_blockers": ["terms_not_accepted"],
                "cf_blockers": ["manual_terms_or_application_required"],
            },
        ],
    }


def test_stage42_eh_schema_has_required_manual_fields() -> None:
    schema = _schema()

    assert schema["required_fields"] == REQUIRED_FIELDS
    assert "terms_accepted_by_user is false" in schema["hard_blocks"]
    assert "This schema does not grant permission." in schema["non_claims"]


def test_stage42_eh_intake_prioritizes_ucy_and_agent_cannot_fill() -> None:
    rows = _intake_rows(_ef())

    assert rows[0]["dataset_id"] == "ucy_crowd_original"
    assert rows[0]["agent_may_fill"] is False
    assert rows[0]["conversion_ready_now"] is False
    assert all(field in rows[0]["user_confirmation"] for field in REQUIRED_FIELDS)


def test_stage42_eh_gate_preserves_legal_blocker() -> None:
    rows = _intake_rows(_ef())
    payload = {
        "input_reports": {"stage42_ef_gate": {"passed": 13, "total": 13}},
        "schema_written": True,
        "intake_template_written": True,
        "user_action_required_written": True,
        "intake_rows": rows,
        "summary": _summary(rows, _ef()),
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_eh_source_terms_confirmation_intake_pass"
