from itertools import combinations

import numpy as np
import pytest

from src.evaluation.m3w_angle_budget_match import match, qualified_expectation


def test_match_removes_only_swaps_and_breaks_ties_in_frozen_order():
    keep = np.array([True, True, True, True, False])
    a = np.array([1, 2, 2, 0, 2])
    b = np.array([0, 1, 0, 2, 1])
    score = np.array([.2, .2, .7, .9, 9.])
    k, (ma, mb), pools = match(keep, a, b, score, score)
    assert k == 2
    np.testing.assert_array_equal(ma, [1, 0, 2, 0, 0])
    np.testing.assert_array_equal(mb, [0, 1, 0, 2, 0])
    np.testing.assert_array_equal(pools[0], [0, 1, 2])


def test_random_expected_count_matches_exact_subset_enumeration():
    q = np.array([[False, True, False], [False, False, True], [False, False, False]])
    choice, pool = np.array([1, 2, 1]), np.arange(3)
    counts = [sum(q[i, choice[i]] for i in subset) for subset in combinations(pool, 2)]
    assert qualified_expectation(q, choice, pool, 2) == pytest.approx(np.mean(counts))
    assert qualified_expectation(q, choice, [], 0) == 0.
