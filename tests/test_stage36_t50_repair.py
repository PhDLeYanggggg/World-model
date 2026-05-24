from pathlib import Path

import numpy as np

from src import stage36_t50_repair as s36


def test_stage36_paths_are_isolated() -> None:
    assert s36.OUT_DIR == Path("outputs/stage36_t50_repair")
    assert s36.DATA_DIR == Path("data/stage36_t50_repair")
    assert 50 in s36.HORIZONS
    assert "scene_clamped_baseline" in s36.BASELINES


def test_stage36_metric_from_selection_uses_all_horizons() -> None:
    # This is a light integration check against the cached Stage35 labels.
    labels = s36._labels("test")
    selected = labels["strongest_idx"].astype(int)
    metrics = s36._metric_from_selection("test", selected)
    assert metrics["rows"] == len(selected)
    assert metrics["t50_improvement"] == 0.0
    assert metrics["easy_degradation"] == 0.0


def test_stage36_select_horizon_policy_falls_back_when_max_switch_zero() -> None:
    labels = s36._labels("test")
    pred = labels["relative_y"].astype(float)
    selected, conf, reasons = s36._select_horizon_policy(
        pred,
        "test",
        50,
        {"gain": 0.0, "confidence": 0.0, "max_switch": 0.0, "easy_guard": 0.0, "hard_only": 0.0},
    )
    assert np.array_equal(selected, labels["strongest_idx"].astype(int))
    assert float(conf.sum()) == 0.0
    assert reasons.dtype.kind in {"U", "S", "O"}
