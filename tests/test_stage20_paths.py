from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path, verify_local_paths


def test_stage20_missing_path_is_not_conversion_ready(tmp_path):
    report = _verify_path("sdd", tmp_path / "does_not_exist")
    assert not report["exists"]
    assert not report["conversion_ready"]


def test_stage20_existing_text_path_is_conversion_ready(tmp_path):
    data_dir = tmp_path / "data" / "trajnet_original" / "stanford"
    data_dir.mkdir(parents=True)
    (data_dir / "sample.txt").write_text("0 1 0.0 0.0\n1 1 0.1 0.1\n", encoding="utf-8")
    report = _verify_path("trajnet_full", tmp_path)
    assert report["exists"]
    assert report["conversion_ready"]
    assert report["trajectory_files"] == 1

