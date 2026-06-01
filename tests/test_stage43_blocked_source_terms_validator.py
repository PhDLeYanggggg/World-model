from __future__ import annotations

from src.stage43_blocked_source_terms_validator import (
    _gate,
    _iso8601_like,
    _validate_biwi,
    _validate_dataset_row,
)


def test_iso8601_like_accepts_utc_forms() -> None:
    assert _iso8601_like("2026-06-01T10:00:00+00:00")
    assert _iso8601_like("2026-06-01T10:00:00Z")
    assert not _iso8601_like("")
    assert not _iso8601_like("not a date")


def test_validate_dataset_row_blocks_blank_manual_fields() -> None:
    row = {
        "dataset_name": "Town-Center",
        "local_path": ".",
        "preferred_official_url": "https://example.org/source",
        "official_url_candidates": ["https://example.org/source"],
        "technical_support_candidate": True,
        "source_confidence": "low",
        "support_family": "mot_like_or_external_topdown_support",
        "t50_candidate_rows": 10,
        "t100_candidate_rows": 5,
        "metric_status": "unverified",
        "manual_fields_required": {
            "official_url_confirmed": False,
            "official_terms_url": "",
            "license_name": "",
            "terms_accepted_by_user": False,
            "accepted_by_user": "",
            "accepted_at_utc": "",
            "allowed_use": "",
            "source_identity_confirmed": False,
            "calibration_projection_scope_confirmed": False,
            "conversion_scope_confirmed": False,
            "can_use_for_stage43_support": False,
        },
    }
    validation = _validate_dataset_row(row)
    assert validation["ready_for_guarded_conversion_preflight"] is False
    assert validation["training_allowed_now"] is False
    assert "terms_not_accepted_by_user" in validation["blockers"]
    assert "source_identity_not_confirmed_by_user" in validation["blockers"]
    assert "can_use_for_stage43_support_false" in validation["blockers"]


def test_validate_dataset_row_can_be_ready_for_future_preflight_when_user_fields_are_complete() -> None:
    row = {
        "dataset_name": "Wild-Track",
        "local_path": ".",
        "preferred_official_url": "https://example.org/source",
        "official_url_candidates": ["https://example.org/source"],
        "technical_support_candidate": True,
        "source_confidence": "high",
        "support_family": "mot_like_or_external_topdown_support",
        "metric_status": "dataset_local",
        "manual_fields_required": {
            "official_url_confirmed": True,
            "official_terms_url": "https://example.org/terms",
            "license_name": "research terms",
            "terms_accepted_by_user": True,
            "accepted_by_user": "test-user",
            "accepted_at_utc": "2026-06-01T10:00:00Z",
            "allowed_use": "research_only",
            "source_identity_confirmed": True,
            "calibration_projection_scope_confirmed": True,
            "conversion_scope_confirmed": True,
            "can_use_for_stage43_support": True,
        },
    }
    validation = _validate_dataset_row(row)
    assert validation["ready_for_guarded_conversion_preflight"] is True
    assert validation["conversion_executed_now"] is False
    assert validation["training_allowed_now"] is False


def test_validate_biwi_requires_independent_source_fields() -> None:
    validation = _validate_biwi(
        {
            "biwi_independent_source": {
                "status": "blocked",
                "blockers": ["independent_biwi_like_source_missing"],
                "manual_fields_required": {
                    "new_independent_source_path": "",
                    "official_url_confirmed": False,
                    "terms_accepted_by_user": False,
                    "source_identity_confirmed": False,
                    "heldout_source_disjoint_from_train_val": False,
                },
            }
        }
    )
    assert validation["repair_training_allowed_now"] is False
    assert validation["ready_for_repair_training_preflight"] is False
    assert "independent_biwi_like_source_missing" in validation["blockers"]
    assert "heldout_source_disjoint_from_train_val_not_confirmed" in validation["blockers"]


def test_gate_passes_for_blocked_blank_template_validation() -> None:
    row = {
        "dataset_name": "Town-Center",
        "blockers": ["terms_not_accepted_by_user"],
        "ready_for_guarded_conversion_preflight": False,
        "manual_terms_accepted": False,
    }
    payload = {
        "input_verdicts": {
            "stage43_bf": "stage43_bf_blocked_source_terms_identity_packet_pass",
            "template_source": "fresh_stage43_bf_blocked_source_terms_identity_packet",
        },
        "summary": {
            "datasets_validated": 3,
            "manual_terms_accepted_rows": 0,
            "ready_for_guarded_conversion_preflight_rows": 0,
            "conversion_executed_now": 0,
            "training_allowed_now": 0,
            "evaluated_now": 0,
            "biwi_ready_for_repair_training_preflight": False,
        },
        "validations": [row, dict(row, dataset_name="Wild-Track"), dict(row, dataset_name="PETS-2009-S2L1")],
        "biwi_validation": {"repair_training_allowed_now": False},
        "manifest": {"ready_for_guarded_conversion_preflight": []},
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "validator_is_permission": False,
            "manifest_is_conversion": False,
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "converted_external_support_source": False,
            "blocked_source_repair_success_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage43_bg_blocked_source_terms_validation_pass"
    assert gate["passed"] == gate["total"]
