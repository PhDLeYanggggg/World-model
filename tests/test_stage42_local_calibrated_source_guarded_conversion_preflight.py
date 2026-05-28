from pathlib import Path

from src.stage42_local_calibrated_source_guarded_conversion_preflight import (
    _candidate_preflight,
    _confirmation_template,
    _gate,
    run_stage42_local_calibrated_source_guarded_conversion_preflight,
)


def _row():
    return {
        "dataset_name": "Example",
        "root": "external_data/example",
        "root_exists": True,
        "parseable": True,
        "coordinate_unit": "image_pixel_bbox_bottom_center",
        "metric_status": "calibration_file_present_but_world_projection_not_integrated",
        "calibration_file_count": 1,
        "legal_auto_convert_allowed": False,
        "stats": {"t50_rows": 10, "t100_rows": 5, "agent_tracks": 2},
    }


def test_confirmation_template_never_fills_user_acceptance():
    template = _confirmation_template([_row()])
    row = template["datasets"][0]
    assert template["terms_confirmation_is_currently_absent"] is True
    assert row["terms_accepted_by_user"] is False
    assert row["accepted_by_user"] == ""
    assert row["allowed_use"] == ""


def test_preflight_blocks_conversion_without_terms_and_geometry_audit():
    template = _confirmation_template([_row()])
    preflight = _candidate_preflight(_row(), template["datasets"][0])
    assert preflight["technical_ready_for_guarded_conversion_after_terms"] is True
    assert preflight["conversion_allowed_now"] is False
    assert "terms_not_accepted_by_user" in preflight["legal_blockers"]
    assert "world_projection_not_integrated" in preflight["geometry_blockers"]
    assert "global_metric_coordinates" in preflight["forbidden_claims_now"]


def test_gate_passes_as_preflight_not_conversion():
    payload = run_stage42_local_calibrated_source_guarded_conversion_preflight(refresh_readmes=False)
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_jo_local_calibrated_source_guarded_preflight_pass"
    assert payload["summary"]["conversion_allowed_now"] == []
    assert len(payload["summary"]["technical_ready_after_terms"]) >= 1
    assert Path("outputs/stage42_long_research/local_calibrated_source_guarded_conversion_preflight_stage42.json").exists()
