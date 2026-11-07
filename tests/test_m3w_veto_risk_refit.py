import numpy as np

from src.world_model import m3w_veto_risk_refit as m


def test_prefix_is_computed_before_each_current_query():
    a = np.array([[-2.],[0.],[1.],[0.]])
    w = np.array([0,-1,0,-1])
    groups = [np.array([0,1]),np.array([2,3])]
    v = m.prefix(groups,a,a,w)
    np.testing.assert_array_equal(v,[[0,0],[0,0],[2,2],[2,2]])


def test_reopening_preserves_nonveto_and_frozen_witness():
    x = np.ones((3,1,1))
    w = np.array([[[-2.,-2.]]])
    a = np.array([[1.],[1.],[-1.]])
    slack = np.zeros((3,2))
    allowed = np.array([[True],[True],[False]])
    r = np.array([0,-1,-1])
    aa,ee,release = m.reopen(x,w,np.ones(3),np.full((3,1),2.),a,a,slack,allowed,r)
    np.testing.assert_array_equal(release.ravel(),[False,True,False])
    np.testing.assert_array_equal(aa.ravel(),[1.,-1.,-1.])
    np.testing.assert_array_equal(aa,ee)
