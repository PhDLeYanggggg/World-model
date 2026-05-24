from pathlib import Path

import numpy as np

from src import stage37_t50_history as s37


def test_stage37_paths_and_schema_constants() -> None:
    assert s37.OUT_DIR == Path("outputs/stage37_t50_history")
    assert s37.DATA_DIR == Path("data/stage37_t50_history")
    assert s37.MAX_K == 64
    assert "prototype_goal_directed_baseline" in s37.BASELINE_FAMILY
    assert "straight_continue" in s37.PROTOTYPES


def test_stage37_angle_between_basic_vectors() -> None:
    a = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    b = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    out = s37._angle_between(a, b)
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], np.pi / 2)


def test_stage37_baseline_family_eval_accepts_fallback() -> None:
    # -1 means fallback to Stage35 strongest baseline for every row.
    selected = np.full(len(s37._geo("test")["horizon"]), -1, dtype=np.int16)
    metrics = s37._eval_family_selection("test", selected)
    assert metrics["rows"] == len(selected)
    assert metrics["t50_improvement"] == 0.0
    assert metrics["easy_degradation"] == 0.0
