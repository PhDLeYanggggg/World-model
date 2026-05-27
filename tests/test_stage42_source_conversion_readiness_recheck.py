from __future__ import annotations

from pathlib import Path

from src import stage42_source_conversion_readiness_recheck as ds


def test_scan_target_blocks_raw_path_without_terms(tmp_path: Path) -> None:
    raw = tmp_path / "raw_ucy"
    raw.mkdir()
    (raw / "tracks.txt").write_text("0 1 0.0 0.0\n", encoding="utf-8")
    target = {
        "dataset_id": "ucy_crowd_original",
        "domain": "UCY",
        "official_url": "https://example.org/ucy",
        "role": "test",
        "raw_candidates": [str(raw)],
        "derived_candidates": [],
    }
    row = ds._scan_target(target, {"terms_accepted_by_user": False, "conversion_ready": False})
    assert row["raw_path_found"] is True
    assert row["conversion_ready"] is False
    assert "terms_allowed_use_or_acceptance_not_confirmed" in row["blockers"]
    assert row["parseability_hint"] == "trajectory_like_files_present"


def test_scan_target_does_not_count_derived_cache_as_ready(tmp_path: Path) -> None:
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "features.json").write_text("{}", encoding="utf-8")
    target = {
        "dataset_id": "opentraj_toolkit",
        "domain": "OpenTraj",
        "official_url": "https://example.org/opentraj",
        "role": "test",
        "raw_candidates": [str(tmp_path / "missing")],
        "derived_candidates": [str(derived)],
    }
    row = ds._scan_target(target, None)
    assert row["raw_path_found"] is False
    assert row["derived_cache_found"] is True
    assert row["technical_preflight_possible"] is True
    assert row["conversion_ready"] is False
    assert "derived_cache_found_but_not_counted_as_raw_verified_dataset" in row["blockers"]


def test_gate_passes_honest_readiness_blocker() -> None:
    target_rows = [
        {
            "dataset_id": "ucy_crowd_original",
            "raw_path_summaries": [],
            "derived_cache_found": True,
            "terms_confirmed": False,
            "conversion_ready": False,
            "technical_preflight_possible": True,
        },
        {
            "dataset_id": "stanford_drone_dataset",
            "raw_path_summaries": [],
            "derived_cache_found": True,
            "terms_confirmed": False,
            "conversion_ready": False,
            "technical_preflight_possible": True,
        },
        {
            "dataset_id": "tgsim_diagnostic",
            "raw_path_summaries": [],
            "derived_cache_found": False,
            "terms_confirmed": False,
            "conversion_ready": False,
            "technical_preflight_possible": False,
        },
    ]
    while len(target_rows) < 7:
        target_rows.append(
            {
                "dataset_id": f"x{len(target_rows)}",
                "raw_path_summaries": [],
                "derived_cache_found": False,
                "terms_confirmed": False,
                "conversion_ready": False,
                "technical_preflight_possible": False,
            }
        )
    payload = {
        "target_rows": target_rows,
        "summary": {
            "targets_checked": 7,
            "technical_preflight_possible_targets": 2,
            "user_action_required_targets": 7,
            "conversion_ready_targets": 0,
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = ds._gate(payload)
    assert gate["verdict"] == "stage42_ds_source_conversion_readiness_recheck_pass"
    assert gate["passed"] == gate["total"]

