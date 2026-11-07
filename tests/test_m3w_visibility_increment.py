import numpy as np
import pytest

from scripts.run_m3w_visibility_increment import LEARNED, factorized, rank


def test_factorized_objective_limits():
    gain = np.array([[4., 1.], [4., 1.], [4., 1.]])
    radius = np.array([[2., 3.]]*3)
    np.testing.assert_allclose(factorized(gain, radius, np.array([0., 1., .5])),
                               [[2., 1.], [-2., -3.], [0., -1.]])


def test_rank_never_changes_recipients_or_allows_identity():
    anchor = np.array([-1, 3, 4])
    legal = np.array([[True]*3, [False, True, True], [True]*3])
    scores = np.array([[100., 100., 100.], [100., 2., 1.], [3., 0., 0.]])
    np.testing.assert_array_equal(rank(anchor, legal, scores, 1), [-1, 3, LEARNED[0]])
    np.testing.assert_array_equal(rank(anchor, legal, scores, 2), [-1, LEARNED[1], LEARNED[0]])
    with pytest.raises(ValueError, match='not attainable'):
        rank(anchor, legal, scores, 3)
