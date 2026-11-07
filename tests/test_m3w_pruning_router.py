import numpy as np

from src.world_model.m3w_pruning_router import choose, prefix


def test_prefix_is_unchanged_by_suffix():
    x = np.arange(12.).reshape(3, 4)
    g = [np.array([0]), np.array([1, 2])]
    np.testing.assert_array_equal(prefix(g, x)[:1], prefix(g[:1], x))
    x[1:] *= 500.
    np.testing.assert_array_equal(prefix(g, x)[:1], prefix(g[:1], x))


def test_keep_support_can_select_second_candidate_or_abstain():
    g = [np.array([0, 1]), np.array([2])]
    score = np.array([.9, .5, .8])
    support = np.array([2., 0., 3.])
    route = np.array([1., 1.])
    np.testing.assert_array_equal(choose(g, score, support, route, .1, .5, True),
                                  [False, True, False])
    np.testing.assert_array_equal(choose(g, score, support, route, .1, .5, False),
                                  [True, False, True])
    np.testing.assert_array_equal(choose(g, score, support, np.zeros(2), .1, .5, True), False)
