import numpy as np

from src.world_model import m3w_feasible_teacher as m


def test_equal_count_teacher_supervision_and_infeasible_query():
    u = np.array([[5.],[4.],[3.],[8.],[6.]])
    a = np.array([[5.],[-1.],[-1.],[2.],[3.]])
    e = np.zeros_like(u)
    labels,mask,d = m.examples([np.arange(3),np.arange(3,5)],u,a,e,np.ones_like(u,bool))
    np.testing.assert_array_equal(labels[:,:,0],[[1.5,1.5,0.,0.,0.],[0.,1.5,1.5,0.,0.]])
    np.testing.assert_array_equal(mask.ravel(),[True,True,True,False,False])
    assert d['active_queries']==1 and d['switches']==2 and d['changed_queries']==1
    assert d['utility_all_easy_excess']==[[9.,4.,0.],[7.,-2.,0.]]


def test_observed_positive_harm_and_unknown_radius_not_netted():
    cv = np.zeros((3,12,2))
    bank = cv[:,None].copy()
    bank[...,0] = 2.
    y = cv.copy()
    y[0,:,0],y[2,:,0] = .5,2.
    valid = np.ones((3,12),bool)
    valid[1] = False
    u,a,e = m.actual(bank,cv,y,valid,1.)
    np.testing.assert_allclose(np.column_stack((u,a,e)),[[-1.,.99,.99],[-2.,2.,2.],[2.,-.04,0.]])
