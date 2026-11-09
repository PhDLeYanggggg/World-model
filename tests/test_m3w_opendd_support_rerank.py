import numpy as np
import pytest

from src.evaluation.m3w_opendd_support_rerank import rank_with_support, shuffle_by_query


def test_support_ranking_is_symmetric_and_changes_equal_merit_ties():
    logits = np.array([2., -2., -3.])
    first, second = np.array([1, 0]), np.array([2, 2])
    merit = np.ones(2)
    result = rank_with_support(merit, first, second, logits)
    np.testing.assert_array_equal(result, rank_with_support(merit, second, first, logits))
    assert result.argmax() == 1
    np.testing.assert_allclose(result, 1.-.25*np.logaddexp(0., [2., -2.]))


def test_shuffled_control_preserves_scores_within_each_query():
    x = np.arange(10.)
    counts = np.array([4, 6])
    y = shuffle_by_query(x, counts, [3227, 17, 0])
    np.testing.assert_array_equal(y, shuffle_by_query(x, counts, [3227, 17, 0]))
    np.testing.assert_array_equal(np.sort(y[:4]), x[:4])
    np.testing.assert_array_equal(np.sort(y[4:]), x[4:])
    assert not np.array_equal(x, y)
    with pytest.raises(ValueError):
        shuffle_by_query(x, np.array([4, 5]), 0)
