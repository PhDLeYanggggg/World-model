from pathlib import Path

from src.stage42_live_source_calibration_recheck import _gate, _inspect_path, _summary, _target_row


def test_inspect_path_reports_file_extension(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("1 2 3\n", encoding="utf-8")
    row = _inspect_path(str(file_path))
    assert row["exists"] is True
    assert row["is_file"] is True
    assert row["sample_extensions"] == {".txt": 1}


def test_target_row_does_not_count_local_ucy_as_conversion_ready(tmp_path: Path):
    local = tmp_path / "UCY"
    local.mkdir()
    target = {
        "target_id": "ucy_crowd_original",
        "domain": "UCY",
        "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "data_role": "external",
        "paths": [str(local)],
        "known_status": "local",
        "required_before_new_claim": ["terms confirmation"],
    }
    row = _target_row(
        target,
        {"ucy_crowd_original": [{"action_id": "FW-TERMS-ucy_crowd_original", "next_user_action": "fill terms"}]},
        {},
        {"summary": {"unified_queue_count": 0}, "unified_queue": []},
    )
    assert row["local_path_found"] is True
    assert row["new_conversion_ready_now"] is False
    assert "explicit_terms_or_source_identity_not_confirmed" in row["blockers"]


def test_summary_preserves_zero_new_conversion_ready():
    rows = [
        {"local_path_found": True, "existing_converted_artifact_found": True, "new_conversion_ready_now": False},
        {"local_path_found": True, "existing_converted_artifact_found": False, "new_conversion_ready_now": False},
    ]
    summary = _summary(
        rows,
        {"summary": {"conversion_ready_now": 0, "highest_priority_blocker": "FW-TERMS-ucy_crowd_original"}},
        {"summary": {"unified_queue_count": 0}},
    )
    assert summary["local_path_found_targets"] == 2
    assert summary["existing_converted_artifact_targets"] == 1
    assert summary["new_conversion_ready_targets"] == 0
    assert summary["source_action_conversion_ready_now"] == 0


def test_ga_gate_passes_for_safe_recheck_payload():
    payload = {
        "source": "fresh_stage42_live_source_calibration_recheck",
        "input_status": {
            "data_calibration_exists": True,
            "source_action_exists": True,
            "unified_queue_exists": True,
        },
        "summary": {
            "targets_audited": 7,
            "local_path_found_targets": 6,
            "new_conversion_ready_targets": 0,
            "source_action_conversion_ready_now": 0,
        },
        "target_rows": [{"next_action": "fill terms"} for _ in range(7)],
        "user_action_required_written": True,
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ga_live_source_calibration_recheck_pass"
