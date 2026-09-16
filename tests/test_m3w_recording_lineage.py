from pathlib import Path

import numpy as np
import pytest

from src.evaluation.m3w_recording_lineage import audit_caches, read_track, row_keys


def cache(path, source, old_split, offset=0):
    n = 3
    np.savez(path, source_file=np.array([str(source)] * n),
             old_split=np.array([old_split] * n), horizon=np.full(n, 50),
             dt_frame_step=np.full(n, 50), dataset=np.array(["test_fixture"] * n),
             agent_id=np.arange(n), frame_id=np.arange(n),
             current_xy=np.full((n, 2), offset), future_xy=np.full((n, 2), offset + 1))


def fixture_paths(tmp_path, identical=False, teacher_exposed=False):
    paths = {}
    for i, split in enumerate(("train", "val", "test")):
        src = tmp_path / f"{split}.txt"
        x = 0 if identical else i
        src.write_text(f"0 1 {x} 0\n10 1 {x + 1} 0\n")
        paths[split] = tmp_path / f"cache_{split}.npz"
        cache(paths[split], src, "train" if teacher_exposed else split, x)
    return paths


def test_content_duplicate_detected_despite_distinct_paths(tmp_path):
    p = audit_caches(fixture_paths(tmp_path, identical=True), tmp_path)
    assert not p["recording_boundary_pass"]
    assert p["exact_duplicate_groups"][0]["splits"] == ["test", "train", "val"]
    assert not p["training_allowed_with_unchanged_legacy_caches"]


def test_teacher_exposure_blocks_otherwise_disjoint_sources(tmp_path):
    p = audit_caches(fixture_paths(tmp_path, teacher_exposed=True), tmp_path)
    assert p["recording_boundary_pass"]
    assert p["new_evaluation_rows_from_legacy_teacher_train"] == {"val": 3, "test": 3}
    assert not p["legacy_teacher_boundary_pass"]


def test_missing_input_fails_closed(tmp_path):
    paths = fixture_paths(tmp_path)
    paths["test"].unlink()
    p = audit_caches(paths, tmp_path)
    assert p["missing_inputs"]
    assert not p["training_allowed_with_unchanged_legacy_caches"]


def test_clean_boundary_does_not_claim_full_certification(tmp_path):
    p = audit_caches(fixture_paths(tmp_path), tmp_path)
    assert p["training_allowed_with_unchanged_legacy_caches"]
    assert p["full_no_leakage_certification"] is False


def test_numeric_duplicates_survive_formatting_and_order_change(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("0 2 1 3\n10 2 2 4\n")
    b = tmp_path / "obsmat.txt"
    b.write_text("10.0 2 2.0 0 4.0 0 0 0\n0 2 1.0 0 3.0 0 0 0\n")
    np.testing.assert_array_equal(row_keys(read_track(a)[0]), row_keys(read_track(b)[0]))


def test_row_keys_do_not_merge_coordinate_changes():
    a = row_keys(np.array([[0, 1, 2.0, 3.0]]))
    b = row_keys(np.array([[0, 1, 20.0, 30.0]]))
    assert not np.array_equal(a, b)
    assert len(row_keys(np.zeros((0, 4)))) == 0
    with pytest.raises(ValueError):
        row_keys(np.zeros(4))
