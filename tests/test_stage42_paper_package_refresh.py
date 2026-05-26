from pathlib import Path

from src import stage42_paper_package_refresh as s42ac


def test_stage42ac_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42ac.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42ac_gate_requires_aux_mixed_not_overclaimed() -> None:
    result = {
        "source": "fresh_synthesis",
        "inputs": {
            "paper_package_exists": True,
            "claim_audit_exists": True,
            "row_cache_exists": True,
            "unified_ablation_exists": True,
            "retrained_matrix_exists": True,
            "aux_ablation_exists": True,
        },
        "inputs_loaded": {
            "aux_ablation": {
                "stage42_ab_gate": {"passed": 11, "total": 11},
                "interpretation": {"uniform_aux_positive_claim_allowed": False},
            }
        },
        "paper_file_status": [{"contains_stage42_ac": True} for _ in s42ac.PAPER_FILES],
        "claim_boundary_ok": True,
        "claim_boundary": {"stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42ac._gate(result)
    assert gate["passed"] == gate["total"]


def test_stage42ac_replace_section_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Paper\n\nold\n", encoding="utf-8")
    s42ac._replace_section(path, "MARK", ["new"])
    s42ac._replace_section(path, "MARK", ["newer"])
    text = path.read_text(encoding="utf-8")
    assert text.count("MARK:START") == 1
    assert "newer" in text
    assert "new\n" not in text
