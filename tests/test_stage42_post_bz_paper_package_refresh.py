from __future__ import annotations

from pathlib import Path

from src import stage42_post_bz_paper_package_refresh as ca


def test_replace_section_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("intro\n", encoding="utf-8")
    ca._replace_section(path, "X", ["first"])
    ca._replace_section(path, "X", ["second"])
    text = path.read_text(encoding="utf-8")
    assert "first" not in text
    assert "second" in text
    assert text.count("X:START") == 1


def test_evidence_rows_include_bz_ci() -> None:
    by = {
        "source": "by",
        "stage42_by_gate": {"verdict": "stage42_by_t50_floor_relaxability_repair_pass"},
        "summary": {
            "repaired_t50_slices": ["TrajNet|50", "UCY|50"],
            "global_t50_improvement": 0.2,
            "global_easy_degradation": 0.0,
        },
    }
    bz = {
        "source": "bz",
        "stage42_bz_gate": {"verdict": "stage42_bz_t50_repair_statistical_evidence_pass"},
        "summary": {"bootstrap_n": 3000},
        "target_union_evidence": {
            "bootstrap": {
                "t50": {"low": 0.1, "high": 0.2},
                "hard_failure": {"low": 0.08},
                "easy_degradation": {"high": 0.0},
            }
        },
        "slice_evidence": {
            "TrajNet|50": {
                "source": "fresh",
                "rows": 10,
                "ci_positive_and_easy_safe": True,
                "bootstrap": {"t50": {"low": 0.1, "high": 0.2}, "easy_degradation": {"high": 0.0}},
                "metric": {"switch_rate": 0.5},
            }
        },
    }
    rows = ca._evidence_rows(by, bz)
    assert any("target union t50 CI" in row["evidence"] for row in rows)
    assert any(row["item"] == "Stage42-BZ slice TrajNet|50" for row in rows)


def test_gate_passes_with_bz_refresh() -> None:
    payload = {
        "source": "unit",
        "inputs_loaded": {
            "by": {"stage42_by_gate": {"verdict": "stage42_by_t50_floor_relaxability_repair_pass"}},
            "bz": {
                "stage42_bz_gate": {"verdict": "stage42_bz_t50_repair_statistical_evidence_pass"},
                "summary": {"bootstrap_n": 3000},
                "slice_evidence": {
                    "TrajNet|50": {"ci_positive_and_easy_safe": True},
                    "UCY|50": {"ci_positive_and_easy_safe": True},
                },
            },
        },
        "paper_file_status": [
            {"contains_stage42_ca": True},
            {"contains_stage42_ca": True},
        ],
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = ca._gate(payload)
    assert gate["verdict"] == "stage42_ca_post_bz_paper_package_refresh_pass"
    assert gate["passed"] == gate["total"]
