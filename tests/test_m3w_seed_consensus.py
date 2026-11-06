import numpy as np
import pytest

from src.evaluation.m3w_seed_consensus import scores


def test_unanimity_is_intersection_not_a_risk_ratio_guarantee():
    p = np.array([[[-1., -1.], [-1., -1.]], [[1., -1.], [-1., -1.]], [[-1., -1.], [-1., -1.]]])
    r = scores(p)
    np.testing.assert_array_equal((r['unanimity']<=0).all(1), (p<=0).all(2).all(0))
    assert (r['mean'][0]<=0).all()
    assert not (r['unanimity'][0]<=0).all()


def test_score_api_does_not_accept_outcome_fields():
    with pytest.raises(TypeError):
        scores(np.zeros((3, 2, 2)), target=np.ones((2, 2)))


def test_require_all_three_seeds():
    with pytest.raises(ValueError):
        scores(np.zeros((2, 8, 2)))
