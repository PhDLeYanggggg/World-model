from pathlib import Path

import numpy as np

from src import stage34_external_geometry as s34


def test_stage34_paths_are_isolated() -> None:
    assert s34.OUT_DIR == Path("outputs/stage34_external_geometry")
    assert s34.DATA_DIR == Path("data/stage34_external_geometry")
    assert "goal_directed_baseline" in s34.BASELINES_V2


def test_stage34_angle_between_is_finite() -> None:
    a = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    b = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    ang = s34._angle_between(a, b)
    assert ang.shape == (2,)
    assert np.isfinite(ang).all()


def test_stage34_select_respects_zero_switch_policy() -> None:
    strong = np.asarray([1, 1, 1])
    pred = np.asarray([[5.0, 4.0], [1.0, 2.0], [3.0, 1.0]])
    selected, conf = s34._select(strong, pred, {"confidence": 0.0, "gain": 0.0, "max_switch_rate": 0.0})
    assert np.array_equal(selected, strong)
    assert float(conf.sum()) == 0.0
