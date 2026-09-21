import numpy as np
from scripts.verify_m3w_native_gain_harm import errors, check_mean, history_summary


def test_independent_distances_preserve_mask_and_unknown():
    p = np.array([[[3., 4.], [0, 2]], [[9, 9], [2, 0]], [[5, 5], [8, 8]]])
    y = np.zeros_like(p)
    mask = np.array([[True, True], [False, True], [False, False]])
    y[~mask] = np.nan
    ade, fde = errors(p, y, mask, np.array([2., 3., 7.]))
    np.testing.assert_allclose(ade, [7, 6, np.nan])
    np.testing.assert_allclose(fde, [4, 6, np.nan])


def test_zero_reference_percentage_not_manufactured():
    check_mean(np.array([.2, .4]), np.zeros(2), np.ones(2, bool),
        dict(rows=2, model_error=.3, reference_error=0, gain_percent=None))


def test_history_summary_uses_only_past_geometry():
    g = np.zeros((2, 476))
    g[:, 16:24] = np.arange(-7, 1)
    g[1, :16] = np.column_stack((np.arange(8), np.zeros(8))).ravel()
    g[1, 230:238] = 1
    out = history_summary(g, np.array([1., 2.]), np.ones(2, bool))
    assert out['stationary_history_rows'] == 1
    assert out['normalized_velocity_change_max'] == 0
    assert out['native_history_path_median'] == 7
    assert out['visible_neighbor_count_mean'] == .5
    assert history_summary(g, np.ones(2), np.zeros(2, bool)) is None
