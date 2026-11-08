import numpy as np

from src.world_model.m3w_signed_bridge_risk import location_correction, replace_bridge_risk, signed_target


def test_signed_bridge_replacement_keeps_gains_and_old_paths():
    x = np.ones((2, 9, 6))
    r = np.full((2, 3, 2), -.1)
    z = replace_bridge_risk(x, r, np.array([2., 4.]), np.full((3, 2), .05))
    np.testing.assert_array_equal(z[:, :6], x[:, :6])
    np.testing.assert_array_equal(z[..., :2], x[..., :2])
    np.testing.assert_allclose(signed_target(z[:, -3:]), np.broadcast_to([-.1, -.2], (3, 2, 2)).transpose(2, 0, 1))


def test_calibration_weights_locations_not_rows_and_excludes_unsupported_actions():
    a = np.zeros((100, 3, 2))
    b = np.full((2, 3, 2), 4.)
    b[1] = 900.
    support = np.ones((2, 3), bool)
    support[1] = False
    c = location_correction([(a, np.ones((100, 3), bool)), (b, support)])
    np.testing.assert_array_equal(c, np.full((3, 2), 2.))
