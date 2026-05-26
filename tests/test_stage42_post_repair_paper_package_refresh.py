from pathlib import Path

from src import stage42_post_repair_paper_package_refresh as s42aj


def test_replace_section_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Paper\n", encoding="utf-8")
    s42aj._replace_section(path, "X", ["hello"])
    s42aj._replace_section(path, "X", ["world"])
    text = path.read_text(encoding="utf-8")
    assert text.count("<!-- X:START -->") == 1
    assert "world" in text
    assert "hello" not in text


def test_stage42aj_gate_requires_all_inputs_and_boundaries() -> None:
    passed_gate = {"passed": 1, "total": 1}
    payload = {
        "source": "test",
        "inputs_loaded": {
            "ad": {"stage42_ad_gate": passed_gate},
            "af": {"stage42_af_gate": passed_gate},
            "ag": {"stage42_ag_gate": passed_gate},
            "ah": {"stage42_ah_gate": passed_gate},
            "ai": {"stage42_ai_gate": passed_gate},
        },
        "paper_file_status": [
            {"contains_stage42_aj": True},
            {"contains_stage42_aj": True},
        ],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42aj._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_aj_post_repair_paper_package_refresh_pass"
