import numpy as np
import pytest

from src.world_model.m3w_recording_history_risk import fit_predict, prefix_masses, source_folds


def test_prefix_proxy_uses_only_frozen_cv_changes_and_observed_positions():
    h = np.zeros((1, 8, 2))
    h[0, :, 0] = [0., 1., 2., 4., 6., 8., 10., 12.]
    choices = np.array([[[0, 1]]])
    p = prefix_masses(h, choices, np.array([[True, False, True, True]]), 100.)
    np.testing.assert_array_equal(p[0, 1], 0.)
    # CV follows the held observed prefix perfectly; OLS4 diverges.
    np.testing.assert_allclose(p[0, 0], [1.75, 0., 0., 0.])


def test_whole_sources_are_excluded_from_prediction_and_residual_fit():
    rng = np.random.default_rng(5)
    x = rng.random((8, 2, 2, 1))
    y = x[..., 0]*.1
    groups = np.repeat(["a", "b", "c", "d"], 2)
    with pytest.raises(ValueError, match="source-disjoint"):
        fit_predict(x, y, groups, np.array([0, 2]), np.array([1]))
    a, u = source_folds(x, y, groups)
    y[:2] = 100.
    b, v = source_folds(x, y, groups)
    np.testing.assert_array_equal(a[:2], b[:2])
    np.testing.assert_array_equal(u[:2], v[:2])
    assert np.all(u >= a)
