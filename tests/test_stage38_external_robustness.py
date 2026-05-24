from pathlib import Path

import numpy as np

from src import stage38_external_robustness as s38


def test_stage38_paths_and_policy_constants() -> None:
    assert s38.OUT_DIR == Path("outputs/stage38_external_robustness")
    assert s38.DATA_DIR == Path("data/stage38_external_robustness")
    assert s38.EPS > 0


def test_stage38_jsonable_numpy_types() -> None:
    payload = {"x": np.asarray([1, 2]), "flag": np.bool_(True), "value": np.float32(1.5)}
    out = s38._jsonable(payload)
    assert out == {"x": [1, 2], "flag": True, "value": 1.5}


def test_stage38_stage37_selection_shapes() -> None:
    test_geo = s38._geo("test")
    selected, confidence = s38._stage37_family_selection("test")
    assert selected.shape[0] == len(test_geo["horizon"])
    assert confidence.shape == selected.shape
    assert np.any(selected >= 0)


def test_stage38_non_test_selection_is_fallback_only() -> None:
    val_geo = s38._geo("val")
    selected, confidence = s38._stage37_family_selection("val")
    assert selected.shape[0] == len(val_geo["horizon"])
    assert np.all(selected == -1)
    assert np.all(confidence == 0)
