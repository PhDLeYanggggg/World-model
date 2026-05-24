from pathlib import Path

import numpy as np

from src import stage31_external_generalization as s31


def test_stage31_paths_are_separated_from_stage30() -> None:
    assert s31.OUT_DIR == Path("outputs/stage31_m3w_external")
    assert s31.FEATURE_DIR == Path("data/stage31_external_feature_store")
    assert s31.LATENT_DIR == Path("data/stage31_external_latent_cache")


def test_stage31_selection_policy_falls_back_when_no_gain() -> None:
    data = {"strongest_idx": np.array([1, 1], dtype=np.int64)}
    pred = np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]], dtype=np.float64)
    selected, conf = s31._select_with_policy(data, pred, {"confidence": 0.0, "gain": 0.0, "max_switch_rate": 0.5})
    assert selected.tolist() == [1, 1]
    assert conf.shape == (2,)
