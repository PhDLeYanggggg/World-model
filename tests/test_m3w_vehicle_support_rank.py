import numpy as np

from scripts import run_m3w_vehicle_support_rank as rank


def test_prefix_score_uses_only_observed_history_and_donor_units():
    h = np.zeros((2, 8, 2))
    h[:, :, 0] = np.arange(8)
    h[1, 7, 1] = 1
    x, p = rank.features(h, .5)
    assert x.shape == (2, 16)
    assert p[0] == 1 and p[1] < 1
    y, q = rank.features(5*h+10, 2.5)
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(p, q)


def test_support_requires_nonzero_observable_cv_error():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[1:, :, 0] = [[.2], [.2], [.4]]
    valid = np.ones((4, 12), dtype=bool)
    valid[2] = False
    np.testing.assert_array_equal(rank.support_label(cv, y, valid, .25), [False, True, False, False])


def test_support_ranks_actor_but_value_still_ranks_its_legal_paths():
    value = np.zeros((3, 2, 6))
    value[..., :2] = np.array([[[4], [1]], [[2], [3]], [[8], [7]]])
    paths = np.arange(3*2*12*2).reshape(3, 2, 12, 2)
    legal = np.array([[True, True], [True, True], [False, False]])
    p = rank.recipient_proposals(value, np.array([.6, .9, 1.]), paths, legal, {'abc': (0, 3)})
    np.testing.assert_array_equal(p['rows'], [1])
    np.testing.assert_array_equal(p['paths'], paths[1:2, 1])
    np.testing.assert_allclose(p['scores'], [2.7])
