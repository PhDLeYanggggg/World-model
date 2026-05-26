from __future__ import annotations

from pathlib import Path

from src import stage42_paper_freeze_candidate_manifest as cz


def _claim_boundary() -> dict[str, bool]:
    return {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def test_status_marks_metadata_caveats_as_candidate_not_final() -> None:
    cx = {"stage42_cx_gate": {"passed": 16, "total": 16}}
    cy = {
        "stage42_cy_gate": {"passed": 11, "total": 11},
        "summary": {"stage42_dirty_files": 9, "stage42_substantive_dirty_files": 0},
    }
    status = cz._status(cx, cy)
    assert status["freeze_status"] == "candidate_with_metadata_caveats"
    assert status["final_immutable_release"] is False


def test_status_blocks_substantive_caveats() -> None:
    cx = {"stage42_cx_gate": {"passed": 16, "total": 16}}
    cy = {
        "stage42_cy_gate": {"passed": 11, "total": 11},
        "summary": {"stage42_dirty_files": 1, "stage42_substantive_dirty_files": 1},
    }
    status = cz._status(cx, cy)
    assert status["freeze_status"] == "not_freeze_ready"
    assert status["final_immutable_release"] is False


def test_file_row_hashes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("stage42 evidence\n", encoding="utf-8")
    row = cz._file_row(path, "paper_file")
    assert row["exists"]
    assert row["size_bytes"] > 0
    assert len(row["sha256"]) == 64
    assert row["role"] == "paper_file"


def test_gate_passes_for_candidate_with_metadata_caveats(tmp_path: Path, monkeypatch) -> None:
    paper_files = [tmp_path / "paper_a.md", tmp_path / "paper_b.md"]
    for path in paper_files:
        path.write_text("paper\n", encoding="utf-8")
    monkeypatch.setattr(cz, "PAPER_FILES", paper_files)
    payload = {
        "freeze_status": {
            "cx_pass": True,
            "cy_pass": True,
            "metadata_caveats": 2,
            "substantive_caveats": 0,
            "freeze_status": "candidate_with_metadata_caveats",
            "final_immutable_release": False,
        },
        "files": [
            {"role": "paper_file", "exists": True, "sha256": "a" * 64},
            {"role": "paper_file", "exists": True, "sha256": "b" * 64},
            {"role": "frozen_runtime_policy_artifact", "exists": True, "sha256": "c" * 64},
            {"role": "frozen_group_consistency_policy_artifact", "exists": True, "sha256": "d" * 64},
        ],
        "claim_boundary": _claim_boundary(),
    }
    gate = cz._gate(payload)
    assert gate["verdict"] == "stage42_cz_paper_freeze_candidate_manifest_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_metadata_caveat_called_final_release(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("paper\n", encoding="utf-8")
    monkeypatch.setattr(cz, "PAPER_FILES", [paper])
    payload = {
        "freeze_status": {
            "cx_pass": True,
            "cy_pass": True,
            "metadata_caveats": 1,
            "substantive_caveats": 0,
            "freeze_status": "candidate_with_metadata_caveats",
            "final_immutable_release": True,
        },
        "files": [
            {"role": "paper_file", "exists": True, "sha256": "a" * 64},
            {"role": "frozen_runtime_policy_artifact", "exists": True, "sha256": "b" * 64},
            {"role": "frozen_group_consistency_policy_artifact", "exists": True, "sha256": "c" * 64},
        ],
        "claim_boundary": _claim_boundary(),
    }
    gate = cz._gate(payload)
    assert gate["passed"] < gate["total"]
    assert not gate["gates"]["metadata_caveat_not_called_final_release"]
