import numpy as np

from src.evaluation.m3w_fixed_roster_shape_capacity import assess
from src.world_model import m3w_radius_matched_turn_menu as m


def test_past_only_menu_matches_peak_and_rigid_transform():
    t = np.arange(8, dtype=float)
    h = np.stack((np.c_[t, .07*t*t], np.c_[2*t, -t], np.c_[t, .04*t*t]))
    amp = np.array([1., .5, 0.])
    c, _, b = m.make_menu(h, 12, amp)
    radius = m.peak(b-c[:, None])
    np.testing.assert_allclose(radius, np.broadcast_to(radius[:, :1], radius.shape))
    assert radius[0, 0] > 0. and not radius[2].any()
    assert not np.allclose(b[0, 0], b[0, -1])
    rot = np.array([[0., 1.], [-1., 0.]])
    _, _, b2 = m.make_menu(h@rot+[100., -28.], 12, amp)
    np.testing.assert_allclose(b@rot, b2, atol=1e-9)
    assert m.prefix_rank(h, amp, amp > 0.).shape == (3, 3, len(m.NAMES))


def test_shape_solver_can_change_direction_but_not_the_fixed_roster():
    cv = np.zeros((2, 2, 2))
    ols = cv.copy()
    ols[..., 0] = 2.
    y = cv.copy()
    y[..., 0] = .4
    bank = np.zeros((2, 2, 2, 2))
    bank[:, 0, :, 0], bank[:, 1, :, 0] = -.25, .25
    active = np.array([True, False])
    valid = np.ones((2, 2), dtype=bool)
    assert assess(cv, ols, bank[:, :1], y, valid, "r", (1., 2.), active) == "risk_infeasible"
    assert assess(cv, ols, bank, y, valid, "r", (1., 2.), active) == "verified_positive_both"
    assert assess(cv, ols, bank, y, valid, "r", (.2, 2.), active) == "absent_easy"
