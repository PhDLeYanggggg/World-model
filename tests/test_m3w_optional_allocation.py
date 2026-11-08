import numpy as np
import pytest

from scripts.audit_m3w_optional_allocation_ols_limit import max_additive_gain
from src.world_model.m3w_optional_allocation import allocate, choose_query


def test_joint_can_use_complementary_risk_at_the_same_count():
    cv = np.array([[4., 2.], [3., 2.], [4., 0.]])
    ols = np.array([[1., 2.], [3., 2.], [4., 0.]])
    excess = np.array([[[.1, -.2], [.5, .5]], [[-.2, .1], [.5, .5]],
                       [[.2, .2], [.5, .5]]])
    legal = np.ones_like(cv, dtype=bool)
    joint, gain, priced = choose_query(cv, ols, excess, legal, 2)
    np.testing.assert_array_equal(joint, [0, 0, -1])
    assert gain[2] >= 0
    assert np.sum(priced >= 0) == 2
    assert np.sum(gain >= 0) == 2


def test_abstain_if_only_illegal_or_nonpositive_actions():
    cv = np.array([[1., -1.], [2., -2.]])
    excess = np.ones((2, 2, 2))
    legal = np.ones_like(cv, dtype=bool)
    assert (choose_query(cv, cv, excess, legal, 2) == -1).all()
    assert (choose_query(-cv, -cv, -excess, cv > 0, 2) == -1).all()
    with pytest.raises(ValueError):
        choose_query(cv, cv, excess, legal, 3)


def test_query_grouping_prevents_cross_query_risk_cancellation():
    x = np.ones((2, 1))
    excess = np.array([[[.5, -.5]], [[-.5, .5]]])
    legal = np.ones_like(x, bool)
    for inverse, cap, expected in [([0, 0], [2], 2), ([0, 1], [1, 1], 0)]:
        action = allocate(x, x, excess, legal, np.array(inverse), np.array(cap), x, 0.)
        np.testing.assert_array_equal((action >= 0).sum(1), expected)


def test_additive_ceiling_does_not_fill_negative_or_cross_query_cap():
    x = np.array([2., 5., -1., 6., 7.])
    inverse = np.array([0, 0, 1, 1, 1])
    assert max_additive_gain(x, inverse, np.array([1, 3])) == 18.
    assert max_additive_gain(x, inverse, np.array([0, 1])) == 7.
