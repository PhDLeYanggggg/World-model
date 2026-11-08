import numpy as np

from src.world_model.m3w_moving_budget_risk import choose


def test_count_match_counts_agents_not_families():
    scores = np.array([[4., 3.], [1., 2.], [9., 8.]])
    legal = np.array([[1, 1], [1, 1], [0, 0]], bool)
    np.testing.assert_array_equal(choose(scores, legal, 2), [0, 1, -1])
    np.testing.assert_array_equal(choose(scores, legal, 0), [-1, -1, -1])
