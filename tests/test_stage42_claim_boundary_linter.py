from __future__ import annotations

from pathlib import Path

from src import stage42_claim_boundary_linter as fv


def test_boundary_context_allows_negated_foundation_claim() -> None:
    assert fv._is_boundary_context("当前不是 large-scale foundation world model。", []) is True
    assert fv._is_boundary_context("Stage5C latent generative 未执行。", []) is True
    assert fv._is_boundary_context("- foundation world model", [], ["But it still is not:"]) is True


def test_scan_file_flags_positive_foundation_claim(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("# Result\n\nM3W is a foundation world model.\n", encoding="utf-8")

    rows = fv.scan_file(path)

    assert rows
    assert rows[0]["check"] == "foundation"


def test_scan_file_allows_non_claim_section(tmp_path: Path) -> None:
    path = tmp_path / "ok.md"
    path.write_text(
        "# Absolute Non-Claims\n\n- Not a foundation world model.\n- t+50 is not seconds-level.\n",
        encoding="utf-8",
    )

    assert fv.scan_file(path) == []


def test_scan_file_allows_diagnostic_auxiliary_context(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic.md"
    path.write_text(
        "# Result\n\nBoth gates selected control, so neighbor/interaction remains diagnostic rather than main claims.\n",
        encoding="utf-8",
    )

    assert fv.scan_file(path) == []


def test_gate_fails_on_metric_overclaim() -> None:
    payload = {
        "source": fv.SOURCE,
        "summary": {"files_scanned": 10},
        "fu_module_boundary_ok": True,
        "violations": [{"check": "metric_seconds"}],
        "claim_boundary": {"stage5c_executed": False, "smc_enabled": False},
    }

    gate = fv._gate(payload)

    assert gate["gates"]["no_metric_seconds_overclaim"] is False
    assert gate["verdict"] == "stage42_fv_claim_boundary_linter_partial"
