import numpy as np

from scripts.run_m3w_vehicle_shortlist import labels, options, outcomes, screen
from src.world_model.m3w_vehicle_shortlist import order, scores


def test_shortlist_preserves_margin_order_and_distinct_actions():
    a = np.zeros((2, 12, 3))
    a[1, 3, 0], a[0, 5, 0], a[0, 2, 0] = .9, .7, .5
    c, s = order(a)
    np.testing.assert_array_equal(c, [[1, 4], [0, 6], [0, 3]])
    np.testing.assert_allclose(s, [.9, .7, .5])


def test_utility_reward_and_harm_have_opposite_effects():
    z = np.zeros((2, 6))
    z[:, 1:3] = 8.
    z[0, 3] = z[1, 4] = 2.
    strict, utility = scores(z, np.ones(2))
    assert strict[0] == strict[1] and utility[0] > 0. > utility[1]


def test_local_shortlist_matches_exhaustive_strict_criterion():
    cv = np.zeros((3, 12, 2))
    anchor = cv.copy()
    anchor[:2, :, 0] = 1.
    selected = np.array([0, 1])
    shortlist = np.array([[0, 1], [1, 3], [0, 4]])
    p = options(cv, anchor, selected, shortlist)
    y = p[1].copy()
    valid = np.ones((3, 12), dtype=bool)
    valid[0, -1] = False
    w = np.full(3, 1/3)
    cut = (3., 5.)
    cap = .02*screen.oracle.moments(cv, cv, y, valid, cut[0])['ref_easy'].sum()
    out = outcomes(cv, cv, anchor, p, selected, y, valid, w, cut, 0., cap)
    oracle = screen.oracle.legal_wins(cv, cv, y, valid, w, anchor, selected, cut, 0., cap)
    np.testing.assert_array_equal(out[3][1:], oracle[shortlist[:, 0], shortlist[:, 1]-1])
    assert out[3][1]
    z = labels(anchor, p, selected, shortlist, y, valid, out)
    assert z.shape == (3, 6) and np.isfinite(z).all()
    assert z[0, 3] > z[0, 4] > 0.
