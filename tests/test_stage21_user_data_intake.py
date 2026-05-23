from __future__ import annotations

from src.stage21_user_data_intake import _inspect_archive


def test_stage21_archive_inspection_missing_path(tmp_path):
    report = _inspect_archive(tmp_path / "missing.zip")
    assert report["exists"] is False

