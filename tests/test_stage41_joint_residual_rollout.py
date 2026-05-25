import numpy as np

from src import stage41_joint_residual_rollout as jrr


def test_policy_switch_requires_gain_and_low_harm():
    pred = {
        "gain": np.asarray([0.5, -0.1, 0.5, 0.5]),
        "harm": np.asarray([0.1, 0.1, 0.9, 0.1]),
        "uncertainty": np.asarray([0.1, 0.1, 0.1, 0.9]),
        "traj_risk": np.asarray([0.1, 0.1, 0.1, 0.1]),
        "physical": np.asarray([0.9, 0.9, 0.9, 0.9]),
        "future_close": np.asarray([0.1, 0.1, 0.1, 0.1]),
    }
    switch = jrr._policy_switch(
        pred,
        {
            "gain_min": 0.0,
            "harm_max": 0.5,
            "uncertainty_max": 0.5,
            "traj_risk_max": 0.5,
            "physical_min": 0.5,
            "future_close_max": 0.5,
        },
    )
    assert switch.tolist() == [True, False, False, False]


def test_pred_waypoints_adds_residual_to_floor():
    data = {
        "floor_xy": np.asarray([[[1.0, 1.0], [2.0, 2.0]]], dtype=float),
        "labels": {"normalizer": np.asarray([2.0])},
        "waypoint_valid": np.asarray([[True, True]]),
    }
    pred = {"residual_delta": np.asarray([[[0.5, -0.5], [1.0, 0.0]]], dtype=float)}
    xy = jrr._pred_waypoints(pred, data)
    assert np.allclose(xy, [[[2.0, 0.0], [4.0, 2.0]]])


def test_policy_grid_contains_fallback_candidate():
    pred = {
        "gain": np.asarray([0.1, 0.2, 0.3]),
        "harm": np.asarray([0.1, 0.2, 0.3]),
        "traj_risk": np.asarray([0.1, 0.2, 0.3]),
    }
    grid = jrr._policy_grid(pred)
    assert grid[0]["gain_min"] > 1e8
    assert len(grid) > 1
