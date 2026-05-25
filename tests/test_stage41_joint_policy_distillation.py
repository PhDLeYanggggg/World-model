import numpy as np

from src import stage41_joint_policy_distillation as jpd


def test_horizon_features_shape_and_scale():
    out = jpd._horizon_features(np.asarray([10, 25, 50, 100, 7]))
    assert out.shape == (5, 5)
    assert out[0, 0] == 1.0
    assert out[3, 3] == 1.0
    assert np.isclose(out[2, -1], 0.5)
    assert out[4, :4].sum() == 0.0


def test_apply_policy_distiller_only_uses_model_scores():
    data = {
        "floor_ade": np.asarray([1.0, 1.0, 1.0]),
        "floor_fde": np.asarray([1.0, 1.0, 1.0]),
        "neural_ade": np.asarray([0.2, 2.0, 0.4]),
        "neural_fde": np.asarray([0.2, 2.0, 0.4]),
        "base_switch": np.asarray([False, False, False]),
        "domain": np.asarray(["D", "D", "D"]),
        "horizon": np.asarray([50, 50, 50]),
    }
    scores = {
        "switch_prob": np.asarray([0.9, 0.9, 0.4]),
        "gain_pred": np.asarray([0.8, 0.8, 0.8]),
        "harm_prob": np.asarray([0.1, 0.9, 0.1]),
    }
    policy = {
        "mode": "distiller_only",
        "slices": {"D|50": {"switch_min": 0.5, "gain_min": 0.0, "harm_max": 0.5, "max_switch": 1.0}},
    }
    sel, _fde, switch = jpd._apply_policy(scores, data, policy)
    assert switch.tolist() == [True, False, False]
    assert np.allclose(sel, [0.2, 1.0, 1.0])


def test_apply_policy_base_plus_distiller_can_expand_and_guard():
    data = {
        "floor_ade": np.asarray([1.0, 1.0, 1.0]),
        "floor_fde": np.asarray([1.0, 1.0, 1.0]),
        "neural_ade": np.asarray([0.2, 2.0, 0.4]),
        "neural_fde": np.asarray([0.2, 2.0, 0.4]),
        "base_switch": np.asarray([False, True, False]),
        "domain": np.asarray(["D", "D", "D"]),
        "horizon": np.asarray([50, 50, 50]),
    }
    scores = {
        "switch_prob": np.asarray([0.9, 0.9, 0.9]),
        "gain_pred": np.asarray([0.8, -0.2, 0.8]),
        "harm_prob": np.asarray([0.1, 0.95, 0.1]),
    }
    policy = {
        "mode": "base_plus_distiller",
        "slices": {
            "D|50": {
                "switch_min": 0.5,
                "gain_min": 0.0,
                "harm_max": 0.5,
                "max_switch": 1.0,
                "guard_harm_min": 0.8,
                "guard_gain_max": 0.0,
            }
        },
    }
    sel, _fde, switch = jpd._apply_policy(scores, data, policy)
    assert switch.tolist() == [True, False, True]
    assert np.allclose(sel, [0.2, 1.0, 0.4])
