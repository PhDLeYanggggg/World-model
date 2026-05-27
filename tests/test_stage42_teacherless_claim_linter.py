from pathlib import Path

from src import stage42_teacherless_claim_linter as hg
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/teacherless_claim_linter_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hg._build_payload()


def test_stage42_hg_flags_teacherless_floor_removal_overclaim(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("Teacherless global floor-free neural deployment is ready.\n", encoding="utf-8")
    rows = hg.scan_file(path)
    assert rows
    assert rows[0]["check"] in {"teacherless_as_global_floor_free", "floor_free_deployable_overclaim"}


def test_stage42_hg_allows_bounded_teacherless_claim(tmp_path: Path) -> None:
    path = tmp_path / "ok.md"
    path.write_text(
        "Supported wording: teacherless proximity-guarded switch gate with causal floor fallback.\n",
        encoding="utf-8",
    )
    assert hg.scan_file(path) == []


def test_stage42_hg_gate_passes_current_package() -> None:
    payload = _payload()
    gate = payload["stage42_hg_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["violations_total"] == 0
    assert gate["gates"]["hf_contract_loaded"]
    assert gate["gates"]["no_floor_free_deployable_overclaim"]
    assert gate["gates"]["no_ungated_neural_deployable_overclaim"]
    assert gate["gates"]["allowed_phrase_present"]
    assert gate["gates"]["required_floor_phrase_present"]


def test_stage42_hg_claim_boundary_keeps_stage5c_smc_false() -> None:
    payload = _payload()
    boundary = payload["claim_boundary"]
    assert boundary["stage5c_executed"] is False
    assert boundary["smc_enabled"] is False
    assert "causal floor removal" in boundary["blocked_claims"]
