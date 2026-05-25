import numpy as np

from src import stage41_pure_ucy_neural_statistical_evidence as stats


def test_bootstrap_ci_reports_positive_improvement_for_better_selected():
    floor = np.asarray([10.0, 20.0, 30.0, 40.0] * 20)
    selected = floor * 0.9
    mask = np.ones(len(floor), dtype=bool)
    out = stats._bootstrap_ci(selected, floor, mask, n=100, seed=1)
    assert out["low"] > 0.0
    assert out["bootstrap_n"] == 100


def test_selected_endpoint_arrays_respects_bounded_alpha():
    ds = {
        "current_xy": np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        "cand_delta": np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float64),
        "normalizer": np.asarray([1.0, 1.0], dtype=np.float64),
        "future_xy": np.asarray([[2.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "floor_fde": np.asarray([1.0, 0.0], dtype=np.float64),
        "horizon": np.asarray([50, 50]),
        "hard": np.asarray([True, False]),
        "failure": np.asarray([False, False]),
        "easy": np.asarray([False, True]),
        "domain": np.asarray(["unit", "unit"]),
    }
    pred = {
        "endpoint_delta": np.asarray([[3.0, 0.0], [3.0, 0.0]], dtype=np.float64),
        "gain": np.asarray([1.0, 1.0], dtype=np.float64),
        "harm": np.asarray([0.0, 0.0], dtype=np.float64),
        "physical": np.asarray([1.0, 1.0], dtype=np.float64),
    }
    arrays = stats._selected_endpoint_arrays(pred, ds, {"type": "bounded_endpoint_residual", "mode": "all", "alpha": 0.5, "max_switch": 1.0})
    assert arrays["switch"].tolist() == [True, True]
    assert np.allclose(arrays["selected"], [0.0, 1.0])
