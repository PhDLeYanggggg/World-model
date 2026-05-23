from __future__ import annotations

from pathlib import Path

from src.stage23_pipeline import _mode_limits, _parse_mode


def test_stage23_mode_limits_are_bounded():
    assert _mode_limits("quick-plus")["train"] < _mode_limits("medium")["train"]
    assert _mode_limits("quick-plus")["test"] <= 12000


def test_stage23_parse_mode_defaults_quick_plus():
    assert _parse_mode([]) == "quick-plus"
    assert _parse_mode(["--quick-plus"]) == "quick-plus"
    assert _parse_mode(["--medium"]) == "medium"


def test_stage23_reports_not_raw_data_paths_are_ignored():
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/stage23_*/" in ignored
