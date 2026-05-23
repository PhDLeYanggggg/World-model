from __future__ import annotations

from pathlib import Path

from src.stage24_pipeline import _baseline_eval_limits, _mode_limits, _parse_mode


def test_stage24_default_mode_is_medium():
    assert _parse_mode([]) == "medium"
    assert _parse_mode(["--medium"]) == "medium"
    assert _parse_mode(["--medium-lite"]) == "medium-lite"


def test_stage24_medium_is_larger_than_medium_lite():
    assert _mode_limits("medium")["train"] > _mode_limits("medium-lite")["train"]
    assert _baseline_eval_limits("medium")["test"] > _baseline_eval_limits("medium-lite")["test"]


def test_stage24_large_data_is_ignored():
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/stage24_*/" in ignored
