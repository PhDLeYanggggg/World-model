from __future__ import annotations

from src.stage42_source_terms_intake_validator_bridge import _gate, _summary


def _cg() -> dict:
    return {
        "input_reports": {
            "confirmation_template_source": "fresh_source_terms_confirmation_intake_from_stage42_ef",
            "confirmation_template_path": "outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json",
            "confirmation_template_format": "stage42_eh_intake",
        },
        "summary": {
            "targets_validated": 5,
            "terms_accepted_targets": 0,
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
        },
        "user_action_required": [{"dataset_id": "ucy_crowd_original"} for _ in range(5)],
    }


def _eh() -> dict:
    return {"stage42_eh_gate": {"verdict": "stage42_eh_source_terms_confirmation_intake_pass"}}


def test_stage42_ei_summary_records_eh_intake_validator_path() -> None:
    summary = _summary(_cg(), _eh())

    assert summary["validator_template_format"] == "stage42_eh_intake"
    assert summary["validator_template_path"].endswith("source_terms_confirmation_intake_template_stage42.json")
    assert summary["conversion_ready_targets"] == 0


def test_stage42_ei_gate_passes_only_for_bridge_and_zero_conversion() -> None:
    summary = _summary(_cg(), _eh())
    payload = {
        "summary": summary,
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ei_intake_validator_bridge_pass"
