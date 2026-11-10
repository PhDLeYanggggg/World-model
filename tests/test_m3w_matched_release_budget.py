from itertools import combinations

import numpy as np
import pytest

from src.evaluation import m3w_matched_release_budget as b


def test_selection_matches_both_budgets_and_attains_the_best_feasible_score():
    original = np.array([True, True, False, False, False, False])
    costs = np.array([[3., 3., 1., 5., 3., 3.], [.2, .4, .1, .6, .3, .3]])
    scores = np.array([0., 0., 10., 11., 2., 3.])
    selected = b.select(scores, costs, original)
    target = costs[:, original].sum(1)
    feasible = [ix for ix in combinations(range(len(scores)), 2)
                if np.all(np.abs(costs[:, ix].sum(1)/target-1.) <= b.TOLERANCE)]
    assert scores[selected].sum() == max(scores[list(ix)].sum() for ix in feasible)
    np.testing.assert_array_equal(selected, [False, False, False, False, True, True])


def test_movement_uses_only_distances_and_observed_weights():
    cv = np.zeros((4, 12, 2))
    paths = np.zeros((2, 2, 12, 2))
    paths[..., 0] = 1.
    rows = np.array([[0, 1], [2, 3]])
    np.testing.assert_allclose(b.movement(cv, rows, paths, np.array([.1, .1, .2, .2])),
                               [[24., 24.], [2.4, 4.8]])
    with pytest.raises(ValueError, match='disjoint'):
        b.movement(cv, np.array([[0, 1], [1, 3]]), paths, np.ones(4))
