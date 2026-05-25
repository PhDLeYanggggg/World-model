import numpy as np

from src import stage41_domain_local_full_trajectory_repair as repair


def _labels():
    return {
        "current_xy": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        "normalizer": np.asarray([2.0, 1.0], dtype=np.float64),
        "waypoint_valid": np.ones((2, 4), dtype=bool),
        "waypoint_xy": np.zeros((2, 4, 2), dtype=np.float64),
        "horizon": np.asarray([50, 50], dtype=np.int64),
        "hard": np.asarray([True, False]),
        "failure": np.asarray([False, False]),
        "easy": np.asarray([False, True]),
        "domain": np.asarray(["d", "d"]),
        "candidate_fde": np.ones((2, 3), dtype=np.float64),
    }


def test_endpoint_linearized_mode_uses_predicted_endpoint_only() -> None:
    labels = _labels()
    pred = {
        "waypoint_delta": np.asarray(
            [
                [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 3.0], [4.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.float32,
        )
    }
    out = repair._mode_xy(pred, labels, "endpoint_linearized")
    assert np.allclose(out[0, -1], [2.0, 0.0])
    assert np.allclose(out[0, 1], [1.0, 0.0])
    assert np.allclose(out[1, -1], [1.0, 3.0])
    assert np.allclose(out[1, 2], [1.0, 2.5])


def test_switch_from_params_uses_predictions_not_eval_labels() -> None:
    labels = _labels()
    pred = {
        "traj_risk": np.asarray([0.1, 0.9], dtype=np.float32),
        "physical": np.asarray([0.9, 0.9], dtype=np.float32),
        "interaction": np.asarray([0.0, 0.0], dtype=np.float32),
        "occupancy": np.asarray([0.0, 0.0], dtype=np.float32),
    }
    mask = np.asarray([True, True])
    switch = repair._switch_from_params(pred, labels, mask, {"traj_risk_max": 0.5, "physical_prob_min": 0.5, "max_switch": 1.0})
    assert switch.tolist() == [True, False]
    labels["hard"][:] = False
    labels["easy"][:] = True
    switch_after_label_flip = repair._switch_from_params(pred, labels, mask, {"traj_risk_max": 0.5, "physical_prob_min": 0.5, "max_switch": 1.0})
    assert switch_after_label_flip.tolist() == [True, False]
