import numpy as np

from src import stage41_full_trajectory_world_state as ft


def test_lookup_waypoints_uses_future_only_as_labels():
    track = np.asarray(
        [
            [0, 1, 0.0, 0.0],
            [5, 1, 0.5, 0.0],
            [10, 1, 1.0, 0.0],
            [15, 1, 1.5, 0.0],
            [20, 1, 2.0, 0.0],
        ],
        dtype=float,
    )
    pts, valid = ft._lookup_waypoints(track, frame=0.0, horizon=20, endpoint_xy=np.asarray([2.0, 0.0], dtype=np.float32))
    assert valid.tolist() == [True, True, True, True]
    assert np.allclose(pts[-1], [2.0, 0.0])


def test_trajectory_errors_mask_invalid_waypoints():
    labels = {
        "waypoint_xy": np.asarray([[[1.0, 0.0], [2.0, 0.0]]], dtype=float),
        "waypoint_valid": np.asarray([[True, False]]),
    }
    pred = np.asarray([[[2.0, 0.0], [100.0, 0.0]]], dtype=float)
    ade, fde = ft._trajectory_errors(pred, labels)
    assert np.allclose(ade, [1.0])
    assert np.allclose(fde, [98.0])


def test_binary_metrics_handles_single_class():
    out = ft._binary_metrics(np.asarray([0.1, 0.2]), np.asarray([False, False]))
    assert out["auroc"] is None
    assert out["positive_rate"] == 0.0
