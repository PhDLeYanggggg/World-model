import numpy as np

from src.world_model import m3w_expanded_risk_support as m


def test_later_query_can_consider_an_option_funded_by_original_prefix():
    a = np.array([[-2.],[0.],[1.],[0.]])
    e = a.copy()
    w = np.array([0,-1,-1,0])
    ok = np.ones_like(a,bool)
    actual = np.array([[0.],[0.],[-2.],[3.]])
    groups = [np.array([0,1]),np.array([2,3])]
    out = m.bound(groups,a,e,ok,w,actual,actual,np.array([0.,0.,10.,0.]))
    assert out['minimum_excess']==[-2.,-2.]
    assert out['maximum_easy_reference']==10.
    assert not out['failed_necessary_conditions']


def test_policy_score_uses_prespecified_reference_or_excess_not_ground_truth():
    u = np.array([[2.,3.]])
    e = np.array([[1.,0.]])
    r = np.array([[4.,0.]])
    orig,support,caution = m.score_arms(u,e,r,np.array([2.]))
    np.testing.assert_allclose(orig,[[1.,1.5]])
    np.testing.assert_allclose(support,[[1.5,1.5]])
    np.testing.assert_allclose(caution,[[.875,1.5]])
