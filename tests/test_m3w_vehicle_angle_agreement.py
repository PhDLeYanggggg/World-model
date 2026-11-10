import numpy as np
import pytest

from src.world_model.m3w_vehicle_angle_agreement import choices, threshold


def test_angle_requires_both_positive_gains_and_no_risk_loss():
    z = np.zeros((3, 3, 6))
    z[:, 1, 3] = 3.
    z[:, 1, 5] = -1.
    z[:, 2, 3] = 1.
    z[:, 2, 5] = 2.
    z[1, 2, 1] = -1.
    z[2, 2, 4] = 2.
    x, y, s = choices(z, np.ones((3, 3)), np.full(3, 4))
    np.testing.assert_array_equal(x, [1, 1, 1])
    np.testing.assert_array_equal(y, [2, 0, 0])
    assert s[0] > 0. and s[1] == s[2] == 0.


def test_calibration_stops_before_first_failing_tie_and_never_uses_zero():
    assert threshold(np.array([3., 2., 2., 1., 0.]), np.array([True, True, False, True, True])) == (3., 1)
    assert threshold(np.array([3., 0.]), np.array([False, True])) == (None, 0)
    assert threshold(np.array([0.]), np.array([True])) == (None, 0)
    with pytest.raises(ValueError):
        threshold(np.array([np.nan]), np.array([True]))
