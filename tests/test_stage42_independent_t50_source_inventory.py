from __future__ import annotations

from pathlib import Path

from src import stage42_independent_t50_source_inventory as cc


def test_recommend_excludes_stanford_derived() -> None:
    row = {
        "parsed_rows": 100,
        "stanford_or_sdd_derived": True,
        "t50_capable": True,
        "dataset_family": "TrajNet",
    }
    status, step = cc._recommend(row)
    assert status == "diagnostic_sdd_or_stanford_derived"
    assert "do_not_count" in step


def test_recommend_excludes_synthetic_diagnostic() -> None:
    row = {
        "path": "/tmp/synth_data/orca_circle_crossing_5ped.ndjson",
        "parsed_rows": 100,
        "stanford_or_sdd_derived": False,
        "t50_capable": True,
        "dataset_family": "TrajNet",
    }
    status, step = cc._recommend(row)
    assert status == "diagnostic_or_simulation_only"
    assert "do_not_count" in step


def test_summarize_counts_t50_windows(tmp_path: Path) -> None:
    path = tmp_path / "students001.txt"
    lines = [f"{i * 10} 1 {i}.0 {i}.0\n" for i in range(60)]
    path.write_text("".join(lines), encoding="utf-8")
    row = cc._parse_track_file(path)
    assert row["t50_capable"]
    assert row["estimated_windows"]["t50"] == 10
    assert row["estimated_windows"]["t100"] == 0


def test_gate_passes_with_user_action() -> None:
    payload = {
        "source": "unit",
        "input_reports": {"stage42_cb_verdict": "stage42_cb_t50_source_robustness_pass_with_source_diversity_limit"},
        "summary": {"scanned_files": 1, "t50_capable_files": 1},
        "claim_boundary": {"inventory_counted_as_converted": False, "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "blocker_summary": "source diversity remains limited",
        "user_actions": [{"action": "provide source"}],
        "no_leakage": {"future_endpoint_input": False, "test_threshold_tuning": False},
    }
    gate = cc._gate(payload)
    assert gate["verdict"] == "stage42_cc_independent_t50_source_inventory_pass"
    assert gate["passed"] == gate["total"]


def test_gate_blocks_inventory_as_conversion() -> None:
    payload = {
        "source": "unit",
        "input_reports": {"stage42_cb_verdict": "stage42_cb_t50_source_robustness_pass_with_source_diversity_limit"},
        "summary": {"scanned_files": 1, "t50_capable_files": 1},
        "claim_boundary": {"inventory_counted_as_converted": True, "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "blocker_summary": "source diversity remains limited",
        "user_actions": [{"action": "provide source"}],
        "no_leakage": {"future_endpoint_input": False, "test_threshold_tuning": False},
    }
    gate = cc._gate(payload)
    assert gate["verdict"] == "stage42_cc_independent_t50_source_inventory_partial"
    assert not gate["gates"]["candidate_status_not_counted_as_converted"]


def test_annotate_usage_marks_same_parent_alternate(tmp_path: Path) -> None:
    parent = tmp_path / "students03"
    parent.mkdir()
    used = parent / "obsmat.txt"
    candidate = parent / "obsmat_px.txt"
    row = {
        "path": str(candidate),
        "recommended_status": "candidate_t50_independent_source",
        "recommended_next_step": "eligible",
    }
    out = cc._annotate_usage([row], {str(used): {"splits": ["train"]}})
    assert out[0]["final_status"] == "alternate_representation_of_current_source"
