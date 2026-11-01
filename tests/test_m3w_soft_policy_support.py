import itertools
import numpy as np
import pytest
from scipy.special import softmax
from src.evaluation.m3w_soft_policy_support import minimum_switch_probability,best_case_risk


def test_minimum_switch_probability_is_achieved_at_all_lower_logits():
    cap=np.array([[1.,2.,3.,4.,5.,6.,7.],[0.,0.,0.,0.,0.,0.,0.]])
    ok=np.ones_like(cap,bool);ok[1]=False
    p=minimum_switch_probability(cap,ok,1.)
    z=np.r_[0.,-cap[0]];assert p[0]==pytest.approx(1-softmax(z)[0],rel=1e-14)
    assert p[1]==0


@pytest.mark.parametrize('seed',range(10))
def test_box_risk_bound_matches_exhaustive128_vertices(seed):
    rng=np.random.default_rng(seed);cap=rng.uniform(0,5,(2,7));ok=rng.random((2,7))>.25
    y=rng.normal(size=(2,7,3));t=.7
    value=best_case_risk(y,cap,ok,t)
    expected=np.full((2,3),np.inf)
    for signs in itertools.product((-1,1),repeat=7):
        z=np.where(ok,cap*np.asarray(signs)/t,-np.inf)
        p=softmax(np.column_stack((np.zeros(2),z)),axis=1)[:,1:]
        expected=np.minimum(expected,(p[...,None]*y).sum(1))
    np.testing.assert_allclose(value,expected,rtol=1e-12,atol=1e-12)


def test_strictly_positive_cost_on_all_available_actions_cannot_reach_zero():
    cap=np.ones((1,7));ok=np.ones_like(cap,bool);y=np.ones((1,7,3))
    value=best_case_risk(y,cap,ok,1.)
    np.testing.assert_allclose(value,minimum_switch_probability(cap,ok,1.)[:,None]*np.ones((1,3)),rtol=1e-14)
    assert (value>0).all()


def test_removed_actions_do_not_change_constraint_bound():
    cap=np.ones((1,7));ok=np.zeros_like(cap,bool);y=np.ones((1,7,3))*1e6
    np.testing.assert_array_equal(best_case_risk(y,cap,ok,1.),np.zeros((1,3)))
