from pathlib import Path

import numpy as np

from src import stage35_selective_transfer as s35


def test_stage35_paths_are_isolated() -> None:
    assert s35.OUT_DIR == Path("outputs/stage35_selective_transfer")
    assert s35.DATA_DIR == Path("data/stage35_selective_transfer")
    assert "goal_directed_baseline" in s35.BASELINES


def test_stage35_four_col_reader_handles_basic_file(tmp_path: Path) -> None:
    p = tmp_path / "tiny.txt"
    p.write_text("0 1 1.0 2.0\n10 1 2.0 3.0\n", encoding="utf-8")
    arr = s35._read_four_col(p)
    assert arr.shape == (2, 4)
    assert np.isfinite(arr).all()


def test_stage35_select_zero_switch_policy() -> None:
    pred = np.asarray([[1.0, 0.5], [2.0, 1.0]])
    labels = {"strongest_idx": np.asarray([0, 0])}
    selected, conf = s35._select(pred, labels, {"gain": 0.0, "confidence": 0.0, "risk": 0.0, "easy_block": 1.0, "max_switch": 0.0}, np.ones(2), np.ones(2), np.zeros(2))
    assert np.array_equal(selected, labels["strongest_idx"])
    assert float(conf.sum()) == 0.0
