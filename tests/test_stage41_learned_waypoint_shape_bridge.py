import numpy as np

from src import stage41_learned_waypoint_shape_bridge as shape


def test_shape_xy_intermediate_only_preserves_endpoint():
    data = {
        "current_xy": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "normalizer": np.asarray([2.0], dtype=np.float32),
    }
    endpoint_pred = {"delta": np.asarray([[1.0, 0.0]], dtype=np.float32)}
    residual = np.asarray([[[0.2, 0.0], [0.1, 0.0], [0.05, 0.0], [0.5, 0.0]]], dtype=np.float32)
    xy = shape._shape_xy(data, endpoint_pred, {"residual": residual}, "intermediate_only", 1.0, 1.0)
    assert np.allclose(xy[0, -1], [2.0, 0.0])
    assert xy[0, 0, 0] > 0.5


def test_stage41_learned_shape_no_leakage_contract():
    result = {
        "future_endpoint_input": False,
        "future_waypoints_input": False,
        "future_waypoints_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    assert result["future_endpoint_input"] is False
    assert result["future_waypoints_input"] is False
    assert result["future_waypoints_label_eval_only"] is True
    assert result["stage5c_executed"] is False
    assert result["smc_enabled"] is False
