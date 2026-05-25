import numpy as np

from src import stage41_endpoint_to_full_trajectory_repair as bridge


def test_linear_waypoints_from_endpoint_delta_uses_normalizer() -> None:
    data = {
        "current_xy": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "normalizer": np.asarray([2.0], dtype=np.float64),
    }
    out = bridge._linear_waypoints_from_delta(data, np.asarray([[1.0, 0.0]], dtype=np.float64))
    assert out.shape == (1, 4, 2)
    assert np.allclose(out[0, -1], [3.0, 2.0])
    assert np.allclose(out[0, 1], [2.0, 2.0])


def test_endpoint_policy_variant_blocks_non_t50_horizons() -> None:
    data = {
        "horizon": np.asarray([10, 50], dtype=np.int16),
        "cand_delta": np.zeros((2, 1, 2), dtype=np.float64),
    }
    pred = {"delta": np.ones((2, 2), dtype=np.float64), "uncertainty": np.zeros(2)}
    gate_pred = {"pred_gain": np.ones(2), "pred_harm": np.zeros(2)}
    policy = {"gain_min": 0.0, "harm_max": 0.0, "uncertainty_max": 1.0, "alpha": 1.0}
    selected, switch = bridge._apply_endpoint_policy(data, pred, gate_pred, policy, {50})
    assert switch.tolist() == [False, True]
    assert np.allclose(selected[0], [0.0, 0.0])
    assert np.allclose(selected[1], [1.0, 1.0])
