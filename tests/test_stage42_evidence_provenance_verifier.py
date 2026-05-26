from __future__ import annotations

from pathlib import Path

from src import stage42_evidence_provenance_verifier as cx


def test_source_label_classification() -> None:
    assert cx._source_label("fresh_run_from_cache") == "fresh_run"
    assert cx._source_label("cached_verified_from_prior_stage") == "cached_verified"
    assert cx._source_label("not_run_missing_terms") == "not_run"
    assert cx._source_label("manual") == "unknown_source_label"


def test_gate_passes_with_recorded_dirty_caveats() -> None:
    payload = {
        "artifact_rows": [
            {
                "claim_area": "batch_runtime_replay",
                "json_exists": True,
                "md_exists": True,
                "runner_exists": True,
                "source_label": "fresh_run",
                "gate": {"all_passed": True, "passed": 1, "total": 1},
                "worktree_caveat": True,
            },
            {
                "claim_area": "runtime_replay_paper_refresh",
                "json_exists": True,
                "md_exists": True,
                "runner_exists": True,
                "source_label": "fresh_run",
                "gate": {"all_passed": True, "passed": 1, "total": 1},
                "worktree_caveat": False,
            },
        ],
        "paper_file_status": [
            {
                "exists": True,
                "has_claim_boundary": True,
                "worktree_caveat": True,
            }
        ],
        "summary": {"artifacts_with_worktree_caveat": 1},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cx._gate(payload)
    assert gate["verdict"] == "stage42_cx_evidence_provenance_pass"
    assert gate["passed"] == gate["total"]


def test_gate_from_md_accepts_gates_keyword(tmp_path: Path) -> None:
    gate = tmp_path / "gate.md"
    gate.write_text("- gates: `7 / 7`\n\nVerdict: ok\n", encoding="utf-8")
    parsed = cx._gate_from_md(gate)
    assert parsed is not None
    assert parsed["passed"] == 7
    assert parsed["total"] == 7
    assert parsed["all_passed"]


def test_refresh_reproducibility_inserts_single_section(tmp_path: Path, monkeypatch) -> None:
    repro = tmp_path / "reproducibility_stage42.md"
    repro.write_text("# Repro\n", encoding="utf-8")
    monkeypatch.setattr(cx, "OUT_DIR", tmp_path)
    payload = {
        "artifact_rows": [
            {
                "claim_area": "x",
                "source_label": "fresh_run",
                "gate": {"passed": 2, "total": 2},
                "runner": "run_x.py",
                "worktree_caveat": False,
            }
        ]
    }
    cx._refresh_reproducibility(payload)
    cx._refresh_reproducibility(payload)
    text = repro.read_text(encoding="utf-8")
    assert "Stage42-CX Evidence Provenance / Command Matrix" in text
    assert text.count("STAGE42_CX_EVIDENCE_PROVENANCE:START") == 1
