import numpy as np

from src.models.final_fallback import FallbackSelector


def test_final_fallback_returns_baseline_when_forced():
    selector = FallbackSelector(force_dataset_baseline=True)
    baseline = np.array([[1.0, 2.0]])
    learned = np.array([[5.0, 6.0]])
    out = selector.select(
        baseline,
        learned,
        alpha=np.array([1.0]),
        failure_probability=np.array([1.0]),
        residual_norm=np.array([0.1]),
        metadata={},
    )
    assert np.allclose(out["final_prediction"], baseline)
    assert out["fallback_reason"][0] == "dataset_validation_prefers_strongest_baseline"

