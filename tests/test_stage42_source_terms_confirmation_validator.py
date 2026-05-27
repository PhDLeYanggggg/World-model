from __future__ import annotations

from pathlib import Path

from src.stage42_source_terms_confirmation_validator import _confirmation_by_id, _gate, _validate_confirmation


def _decision(**overrides):
    row = {
        "id": "new_source",
        "name": "New Source",
        "official_url": "https://example.edu/source",
        "blockers": [],
    }
    row.update(overrides)
    return row


def _confirmation(path: Path, **overrides):
    row = {
        "dataset_id": "new_source",
        "official_url": "https://example.edu/source",
        "terms_accepted_by_user": True,
        "terms_acceptance_date": "2026-05-26",
        "allowed_use": "research",
        "local_path": str(path),
        "source_identity": "independent_scene_family",
        "notes": "",
    }
    row.update(overrides)
    return row


def test_blank_confirmation_blocks_conversion():
    validation = _validate_confirmation(_decision(), None)
    assert validation["conversion_ready"] is False
    assert "confirmation_entry_missing" in validation["confirmation_blockers"]


def test_terms_accepted_is_insufficient_when_cf_blocker_remains(tmp_path):
    validation = _validate_confirmation(
        _decision(blockers=["no_independent_t50_candidate"]),
        _confirmation(tmp_path),
    )
    assert validation["terms_accepted_by_user"] is True
    assert validation["conversion_ready"] is False
    assert validation["cf_blockers"] == ["no_independent_t50_candidate"]


def test_ready_requires_terms_path_allowed_use_and_source_identity(tmp_path):
    local = tmp_path / "tracks"
    local.mkdir()
    validation = _validate_confirmation(_decision(), _confirmation(local))
    assert validation["confirmation_blockers"] == []
    assert validation["cf_blockers"] == []
    assert validation["conversion_ready"] is True


def test_eh_nested_intake_confirmation_is_normalized(tmp_path):
    local = tmp_path / "ucy"
    local.mkdir()
    template = {
        "source": "fresh_source_terms_confirmation_intake_from_stage42_ef",
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "official_url_from_prior_audit": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                "user_confirmation": {
                    "terms_accepted_by_user": True,
                    "terms_acceptance_date": "2026-05-27",
                    "official_terms_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                    "allowed_use": "research_only",
                    "local_path": str(local),
                    "source_identity": "UCY_zara01_zara02_students03",
                },
            }
        ],
    }

    confirmations = _confirmation_by_id(template)
    validation = _validate_confirmation(
        _decision(
            id="ucy_crowd_original",
            official_url="https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        ),
        confirmations["ucy_crowd_original"],
    )

    assert validation["confirmation_blockers"] == []
    assert validation["terms_accepted_by_user"] is True
    assert validation["conversion_ready"] is True


def test_gate_requires_zero_ready_for_current_blank_template():
    payload = {
        "source": "fresh_stage42_cg_source_terms_confirmation_validator",
        "input_reports": {
            "stage42_cf_verdict": "stage42_cf_source_conversion_legal_gate_pass",
            "confirmation_template_source": "fresh_stage42_cf_source_conversion_legal_gate",
        },
        "summary": {
            "targets_validated": 5,
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
        },
        "manifest": {"blocked_targets": [1], "conversion_ready_targets": []},
        "user_action_required": [{"dataset_id": "x"}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_cg_source_terms_confirmation_validator_pass"
    assert gate["passed"] == gate["total"]


def test_gate_accepts_stage42_eh_intake_template_source():
    payload = {
        "source": "fresh_stage42_cg_source_terms_confirmation_validator",
        "input_reports": {
            "stage42_cf_verdict": "stage42_cf_source_conversion_legal_gate_pass",
            "confirmation_template_source": "fresh_source_terms_confirmation_intake_from_stage42_ef",
        },
        "summary": {
            "targets_validated": 5,
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
        },
        "manifest": {"blocked_targets": [1], "conversion_ready_targets": []},
        "user_action_required": [{"dataset_id": "x"}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["gates"]["confirmation_template_loaded"] is True
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_blank_template_somehow_claims_ready():
    payload = {
        "source": "fresh_stage42_cg_source_terms_confirmation_validator",
        "input_reports": {
            "stage42_cf_verdict": "stage42_cf_source_conversion_legal_gate_pass",
            "confirmation_template_source": "fresh_stage42_cf_source_conversion_legal_gate",
        },
        "summary": {
            "targets_validated": 5,
            "conversion_ready_targets": 1,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
        },
        "manifest": {"blocked_targets": [], "conversion_ready_targets": [1]},
        "user_action_required": [],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["gates"]["empty_template_blocks_conversion"] is False
    assert gate["verdict"] == "stage42_cg_source_terms_confirmation_validator_partial"
