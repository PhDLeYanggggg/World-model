import numpy as np

from scripts import run_m3w_cv_pooled_coverage as study
from scripts import audit_m3w_cv_uncapped_dual_certificate as cert


def test_pool_can_spend_support_but_not_overdraw_either_budget():
    gain = np.array([[1.,3.],[8.,7.],[5.,4.]])
    cost = np.array([[[-3.,-2.],[1.,0.]],[[2.,3.],[2.,1.]],[[2.,0.],[0.,2.]]])
    easy = np.array([[.4,.4],[0.,0.],[0.,0.]])
    legal = np.ones_like(gain,dtype=bool)
    p = study.pool_query(gain,cost,easy,legal,3,.1)
    np.testing.assert_array_equal(p,[0,1,-1])
    assert (cost[np.arange(2),p[:2]].sum(0)<=0.).all()
    p = study.pool_query(gain,cost,easy,legal,1,.1)
    np.testing.assert_array_equal(p,[0,-1,-1])


def test_no_support_or_nonpositive_pool_falls_back():
    score = np.array([[-2.],[1.]])
    cost = np.array([[[-2.,-2.]],[[1.,1.]]])
    legal = np.ones_like(score,dtype=bool)
    for easy in (np.array([[.5],[.1]]),np.zeros_like(score)):
        np.testing.assert_array_equal(study.pool_query(score,cost,easy,legal,2,.25),[-1,-1])


def test_gain_control_matches_query_count_without_illegal_choices():
    g = np.array([[1.,2.],[4.,3.],[1.,2.]])
    legal = np.array([[True,False],[False,True],[True,True]])
    groups = [np.arange(2),np.array([2])]
    np.testing.assert_array_equal(study.independent(g,legal,groups,match=np.array([0,-1,-1])),[-1,1,-1])


def test_separable_dual_is_an_upper_bound_with_fallback_and_one_path_per_row():
    d = dict(row=np.array([0,0,1]),gain=np.array([[3.,4.],[5.,2.],[1.,1.]]),
             offset=np.array([0.,-1.]),risk=np.array([[0.,0.,0.],[3.,3.,0.],[-1.,-1.,0.]]),
             easy=np.array([True,True,False]),easy_cap=0.)
    # Only the first action can provide the required easy reference at feasible risk.
    expected = 4.
    fn = cert.oracle(d)
    for theta in np.linspace(0.,1.,5):
        for a in (0.,.1,1.,10.):
            value,_,_ = fn(np.r_[theta,a,a,a,a])
            assert value>=expected-1e-9
