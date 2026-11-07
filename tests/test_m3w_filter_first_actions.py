import numpy as np

from scripts.run_m3w_filter_first_actions import select


def test_filter_precedes_ranking_and_preserves_query_cap():
    u = np.array([[6., 3.], [5., 2.], [4., 1.], [9., 7.]])
    a = np.zeros_like(u)
    e = np.array([[1., -1.], [1., -1.], [-1., -1.], [-1., -1.]])
    legal = np.ones_like(u, dtype=bool)
    c = select([np.arange(3), np.array([3])], u, a, e, np.ones(4), legal)
    np.testing.assert_array_equal(c, [1, -1, 0, -1])


def test_no_admissible_positive_cost_falls_back():
    u = np.array([[-1., 2.], [3., 4.]])
    c = select([np.arange(2)], u, np.ones_like(u), np.zeros_like(u), np.ones(2), np.ones_like(u, bool))
    np.testing.assert_array_equal(c, [-1, -1])
