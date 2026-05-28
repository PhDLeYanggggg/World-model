import numpy as np

from src import stage42_eth_ucy_blocked_source_geometry_support as jj


def test_family_waypoints_interpolates_to_family_endpoint():
    data = {
        "current_x": np.asarray([0.0], dtype=np.float32),
        "current_y": np.asarray([0.0], dtype=np.float32),
        "family_pred": np.asarray([[[10.0, 0.0], [0.0, 10.0]]], dtype=np.float32),
    }
    xy = jj._family_waypoints(data)
    assert xy.shape == (1, 2, len(jj.WAYPOINT_FRAC), 2)
    assert np.allclose(xy[0, 0, -1], [10.0, 0.0])
    assert np.allclose(xy[0, 1, -1], [0.0, 10.0])


def test_select_static_family_policy_keeps_floor_when_val_unsafe():
    data = {
        "horizon": np.asarray([50, 50, 50], dtype=np.int16),
        "hard": np.asarray([False, False, False]),
        "failure": np.asarray([False, False, False]),
        "easy": np.asarray([True, True, True]),
    }
    family_ade = np.asarray([[10.0], [10.0], [10.0]])
    family_fde = family_ade.copy()
    floor_ade = np.asarray([1.0, 1.0, 1.0])
    floor_fde = floor_ade.copy()
    policy = jj._select_static_family_policy(family_ade, family_fde, floor_ade, floor_fde, data, np.asarray([True, True, True]))
    assert policy["slices"] == {}
    assert not policy["switch"].any()
    assert np.allclose(policy["selected_ade"], floor_ade)


def test_distribution_shift_ranks_largest_standardized_gap():
    data = {
        "scale": np.asarray([1.0, 1.0, 10.0]),
        "track_length": np.asarray([1.0, 1.0, 1.0]),
        "dt_frame_step": np.asarray([1.0, 1.0, 1.0]),
        "oracle_margin": np.asarray([0.0, 0.0, 0.0]),
        "history_scalar": np.zeros((3, 2), dtype=np.float32),
    }
    rows = jj._distribution_shift(data, np.asarray([True, True, False]), np.asarray([False, False, True]))
    assert rows[0]["feature"] == "scale"
    assert rows[0]["standardized_mean_gap"] > 1.0
