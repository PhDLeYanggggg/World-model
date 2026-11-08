import numpy as np

from src.world_model import m3w_radius_opening as m


def test_offer_radius_is_frozen_and_prefix_cannot_see_later_queries():
    c = np.zeros((4, 2, 2))
    b = np.zeros((4, 1, 2, 2))
    b[:, 0, :, 0] = np.array([4., .5, 7., 5.])[:, None]
    groups = [np.array([i]) for i in range(4)]
    inc = np.full(4, -1)
    full = m.offers(c, b, groups, inc)
    short = m.offers(c, b, groups[:1], inc)
    assert len(full) == 2 and len(short) == 1
    np.testing.assert_array_equal(full[0][2], short[0][2])
    assert full[1][0][0] == 2 and full[0][3] == full[1][3] == 1.
    np.testing.assert_allclose(full[1][2][..., 0], 1.)
    inc[2] = 0
    assert len(m.offers(c, b, groups, inc)) == 1


def test_choice_does_not_use_future_features_until_waiting():
    a = dict(x=np.array([[1., 1.], [1., 2.]]), radius=1.)
    b = dict(x=np.array([[1., 3.], [1., 4.]]), radius=1.)
    pick = np.array([0., 1.])
    wait = np.zeros(5)
    wait[0] = 1.
    assert m.choose([a], pick, wait) == ((0, 1), None, True)
    assert m.choose([a, b], pick, wait) == ((0, 1), (1, 1), True)
    wait[0] = -1.
    assert m.choose([a, b], pick, wait) == ((0, 1), (0, 1), False)


def test_zero_data_fit_preserves_nonwait_default():
    a, b = m.fit({"site": []}, 2)
    np.testing.assert_array_equal(a, np.zeros(2))
    np.testing.assert_array_equal(b, np.zeros(5))
