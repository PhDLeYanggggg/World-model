import numpy as np

from src import stage41_weighted_pairwise_shape_switch_policy as weighted


def test_training_weight_upweights_positive_gain_and_hard_tail():
    pack = {
        "labels": {
            "horizon": np.asarray([10, 50, 100, 50]),
            "hard": np.asarray([False, True, False, False]),
            "failure": np.asarray([False, False, True, False]),
        },
        "old_shape": {"shape_switch": np.asarray([True, True, False, True])},
    }
    gain = np.asarray([-0.1, 0.2, 0.3, -0.2], dtype=np.float64)
    weight = weighted._training_weight(pack, "old_shape", gain)
    assert weight[1] > weight[0]
    assert weight[2] > weight[0]
    assert weight[1] > weight[3]


def test_weighted_ridge_fit_returns_finite_weights():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    w = weighted._weighted_ridge_fit(x, y, np.asarray([1.0, 1.0, 4.0, 4.0]), lam=0.01)
    assert w.shape == (2,)
    assert np.isfinite(w).all()
