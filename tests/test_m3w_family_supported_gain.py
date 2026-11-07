import numpy as np

from src.world_model.m3w_family_supported_gain import moments, replace


def test_moments_use_supported_positive_gain_and_risk_excess():
    x = np.ones((3, 1, 1))
    a = np.array([[100., 50., 0., 0., -2., 4.],
                  [100., 0., 0., 0., 0., 9.],
                  [100., 50., 3., 2., 2., -1.]])[:, None]
    g, b = moments(x, a, np.ones((3, 1), bool), np.ones(3), np.ones(3))
    np.testing.assert_allclose(g, 1.)
    np.testing.assert_allclose(b, [[[4./3, 2./3]]])


def test_rank_replaces_agents_without_changing_ordered_family_slots():
    q = np.arange(3)
    j = np.array([1, 0, 1])
    x = np.broadcast_to(np.arange(3.)[:, None, None], (3, 2, 1))
    old = np.zeros((3, 13))
    old[:, 9] = np.arange(3)/8
    co = np.array([[[1., 0.]], [[1., 0.]]])
    floor = np.zeros((3, 12, 2))
    bank = np.ones((3, 2, 12, 2))
    legal = np.array([[True, True], [True, True], [False, True]])
    _, i, k, f = replace([(q, q, j, old)], x, np.ones(3), x[..., 0], x[..., 0], x[..., 0],
                         legal, np.zeros((2, 1, 4)), co, bank, floor)[0]
    np.testing.assert_array_equal(i, [2, 1, 1])
    np.testing.assert_array_equal(k, j)
    np.testing.assert_array_equal(f[:, 9], old[:, 9])
