from __future__ import annotations

from pathlib import Path

from src import stage42_raw_source_parseability_dry_run as dt


def test_classify_obsmat_and_homography() -> None:
    obsmat = dt._classify_file(
        Path("obsmat.txt"),
        ["1 2 -2.0 0 3.5 0.1 0 0.2", "11 2 -1.9 0 3.6 0.1 0 0.2"],
    )
    homography = dt._classify_file(Path("H.txt"), ["1 0 0", "0 1 0", "0 0 1"])
    assert obsmat["parser_family"] == "obsmat_like_8col_trajectory"
    assert obsmat["trajectory_like"] is True
    assert homography["parser_family"] == "homography_matrix_candidate"
    assert homography["calibration_like"] is True


def test_summarize_target_is_sample_only_and_not_legal_ready(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "tracks.txt").write_text("0 1 1.0 2.0\n10 1 2.0 3.0\n", encoding="utf-8")
    target = {
        "dataset_id": "trajnetplusplus_official",
        "domain": "TrajNet",
        "conversion_ready": False,
        "raw_path_summaries": [{"path": str(raw), "exists": True}],
    }
    row = dt._summarize_target(target)
    assert row["dry_run_parseable"] is True
    assert row["legal_conversion_ready"] is False
    assert row["conversion_executed"] is False
    assert row["world_state_rows_generated"] == 0


def test_gate_blocks_conversion_overclaim() -> None:
    payload = {
        "stage42_ds_verdict": "stage42_ds_source_conversion_readiness_recheck_pass",
        "summary": {
            "world_state_rows_generated": 10,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "dry_run_parseable_targets": 3,
            "targets_with_homography_or_time_hints": 1,
            "legal_conversion_ready_targets": 0,
            "archives_extracted_now": 0,
            "user_action_required_targets": 5,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = dt._gate(payload)
    assert gate["gates"]["sample_only_no_conversion"] is False
    assert gate["passed"] == gate["total"] - 1

