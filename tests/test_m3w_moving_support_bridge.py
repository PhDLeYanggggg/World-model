import numpy as np

from src.world_model.m3w_moving_support_bridge import best_bridge


def test_bridge_reserves_one_easy_query_instead_of_double_spending_it():
    best = np.array([9., 7., 5., 1.])
    easy = np.array([1., 0., 0., 2.])
    gain, q, rest = best_bridge(best, easy, 2)
    assert (gain, q, rest.tolist()) == (11., 3, [0])
    gain, q, rest = best_bridge(best, easy, 1)
    assert (gain, q, rest.tolist()) == (2., 3, [])
    assert best_bridge(best, np.zeros(4))[1] is None
