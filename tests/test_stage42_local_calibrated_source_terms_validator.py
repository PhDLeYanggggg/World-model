from pathlib import Path

from src.stage42_local_calibrated_source_terms_validator import (
    _gate,
    _validate_row,
    run_stage42_local_calibrated_source_terms_validator,
)


def test_blank_local_terms_template_blocks_conversion():
    payload = run_stage42_local_calibrated_source_terms_validator(refresh_readmes=False)
    assert payload["summary"]["datasets_validated"] >= 3
    assert payload["summary"]["terms_accepted_rows"] == 0
    assert payload["summary"]["conversion_ready_rows"] == 0
    assert payload["summary"]["converted_now"] == 0
    assert payload["summary"]["evaluated_now"] == 0
    for row in payload["validations"]:
        assert row["conversion_ready"] is False
        assert row["converted_now"] is False


def test_validate_row_requires_user_acceptance_and_existing_path(tmp_path: Path):
    prefill = {
        "official_url_candidates": ["https://example.org/dataset"],
        "preferred_official_url": "https://example.org/dataset",
        "source_confidence": "high",
        "technical_ready_after_terms": True,
        "t50_rows": 12,
        "t100_rows": 7,
        "coordinate_unit": "dataset_local",
        "metric_status": "unverified",
    }
    base = {
        "dataset_name": "Example",
        "local_path": str(tmp_path),
        "official_url": "https://example.org/dataset",
        "official_terms_url": "https://example.org/dataset/terms",
        "license_name": "research terms",
        "terms_accepted_by_user": True,
        "accepted_by_user": "user",
        "accepted_at_utc": "2026-05-28T00:00:00Z",
        "allowed_use": "research_only",
        "commercial_use_allowed": False,
        "derived_data_allowed": True,
        "redistribution_allowed": False,
        "source_identity_confirmed": True,
        "conversion_scope_confirmed": True,
    }
    ready = _validate_row(base, prefill)
    assert ready["conversion_ready"] is True
    assert ready["conversion_allowed_now"] is False

    blocked = _validate_row({**base, "terms_accepted_by_user": False}, prefill)
    assert blocked["conversion_ready"] is False
    assert "terms_not_accepted_by_user" in blocked["blockers"]


def test_jq_gate_keeps_conversion_disabled():
    payload = run_stage42_local_calibrated_source_terms_validator(refresh_readmes=False)
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_jq_local_calibrated_source_terms_validation_pass"
    assert gate["gates"]["conversion_allowed_count_zero"] is True
    assert payload["claim_boundary"]["validator_is_permission"] is False
    assert payload["queue"]["conversion_executed"] is False
