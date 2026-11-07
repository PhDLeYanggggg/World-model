import numpy as np
import pytest

from scripts.run_m3w_structured_imitation import choose, paired_moments


def test_two_preference_types_have_equal_weight():
    x = np.array([[[4.], [3.]], [[2.], [0.]]])
    a, b = paired_moments(x, np.ones((2, 2), bool), np.array([0, -1]))
    np.testing.assert_allclose(a, [[5.5]])
    np.testing.assert_allclose(b, [2.])


def test_choose_matches_count_and_rejects_identity_actions():
    score = np.array([[100., 2.], [3., 1.], [8., 9.]])
    legal = np.array([[False, True], [True, True], [False, False]])
    np.testing.assert_array_equal(choose(score, legal, 2), [1, 0, -1])
    with pytest.raises(ValueError, match='Cannot match'):
        choose(score, legal, 3)
