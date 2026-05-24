from pathlib import Path

import numpy as np

from src import stage39_neural_dynamics as s39


def test_stage39_paths_and_limits() -> None:
    assert s39.OUT_DIR == Path("outputs/stage39_neural_dynamics")
    assert s39.DATA_DIR == Path("data/stage39_neural_dynamics")
    assert s39.SEQ_K == 32
    assert s39.THREADS >= 1


def test_stage39_stage37_policy_prediction_shapes() -> None:
    pred, fde, fallback, confidence = s39._stage37_policy_prediction("test")
    assert pred.ndim == 2
    assert pred.shape[1] == 2
    assert fde.shape == fallback.shape == confidence.shape
    assert len(fde) == pred.shape[0]
    assert np.all(np.isfinite(fde))


def test_stage39_metrics_from_fde_preserves_fallback() -> None:
    ds = s39._ds("test") if (s39.DATA_DIR / "neural_dataset_test.npz").exists() else None
    if ds is None:
        return
    fallback = ds["fallback_fde"].astype(float)
    metrics = s39._metrics_from_fde(fallback.copy(), fallback, ds)
    assert metrics["all_improvement"] == 0.0
    assert metrics["easy_degradation"] == 0.0
