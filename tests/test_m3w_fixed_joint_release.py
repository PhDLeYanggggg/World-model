import numpy as np
import pytest

from src.evaluation import m3w_fixed_joint_release as rule


def test_support_score_only_reranks_fixed_pairs():
    raw = np.array([.7, .1])
    p = np.array([.02, .03, .5, .7])
    rows = np.array([[0, 1], [2, 3]])
    score = rule.scores(raw, p, rows)
    np.testing.assert_array_equal(score[0], raw)
    np.testing.assert_allclose(score[2], [.0494, .85])
    assert [s.argmax() for s in score] == [0, 1, 1]
    with pytest.raises(ValueError, match='two distinct'):
        rule.scores(raw, p, np.array([[0, 0], [2, 3]]))


def test_same_budget_at_ties_and_exact_fallback_for_unused_actors():
    np.testing.assert_array_equal(rule.quota(np.zeros(8)), [1, 1, 0, 0, 0, 0, 0, 0])
    cv = np.zeros((5, 12, 2))
    rows = np.array([[0, 3], [1, 4]])
    paths = np.ones((2, 2, 12, 2))
    prediction = rule.prediction(cv, rows, paths, np.array([False, True]))
    np.testing.assert_array_equal(prediction[[1, 4]], paths[1])
    np.testing.assert_array_equal(prediction[[0, 2, 3]], cv[[0, 2, 3]])
