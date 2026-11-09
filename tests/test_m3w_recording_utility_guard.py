import numpy as np
import pytest

from src.world_model.m3w_recording_utility_guard import fit_predict, folds, prefix_utility


def test_prefix_scores_relative_to_ols_on_fixed_choices():
    h = np.zeros((1, 8, 2))
    h[0, :, 0] = [0., 1., 2., 4., 5.3, 6.6, 7.9, 9.2]
    gain, ref = prefix_utility(h, np.array([[0, 1]]))
    assert gain[0] == 0.
    assert gain[1] < 0.
    assert ref == pytest.approx(1.75)


def test_utility_calibration_never_uses_the_held_source():
    x = np.arange(8.)[:, None]
    y = (x[:, 0]-4.)**2
    groups = np.repeat(["a", "b", "c", "d"], 2)
    with pytest.raises(ValueError, match="source-disjoint"):
        fit_predict(x, y, groups, np.array([0, 2]), np.array([1]))
    p, lo = folds(x, y, groups)
    y[:2] = 100.
    q, lower = folds(x, y, groups)
    np.testing.assert_array_equal(p[:2], q[:2])
    np.testing.assert_array_equal(lo[:2], lower[:2])
    assert (lo <= p).all()
