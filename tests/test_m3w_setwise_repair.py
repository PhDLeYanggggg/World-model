import numpy as np

from src.world_model import m3w_setwise_repair as m


def test_slate_uses_same_count_distinct_agents_and_dominates_two_costs():
    u = np.array([[7.,1.],[5.,2.],[4.,3.]])
    a = np.array([[9.,-1.],[1.,0.],[1.,0.]])
    e = np.array([[0.,0.],[2.,1.],[0.,-1.]])
    orig = np.array([1,-1,0])
    z = m.slate(u,a,e,np.ones_like(u,bool),orig)
    np.testing.assert_array_equal(z[0],[[0,1],[2,0]])
    assert len(z)>1 and (z[:,0,0]!=z[:,1,0]).all()
    for cost in (a,e):
        x = cost[z[:,:,0],z[:,:,1]].sum(1)
        assert (x<=x[0]).all()


def test_mixed_feature_vanishes_for_only_one_changed_path():
    cv = np.zeros((2,12,2))
    bank = cv[:,None].copy()
    bank[0,0,:,0] = 1.
    sets = np.array([[[0,0],[1,0]]])
    origin = np.array([[100.,200.],[101.,200.]])
    np.testing.assert_allclose(m.mixed(bank,cv,origin,np.ones(2),sets),0.,atol=1e-12)
    np.testing.assert_allclose(m.mixed(bank,cv,origin,np.ones(2),sets[:,:1]),0.)
    bank[1,0,:,0] = -1.
    np.testing.assert_allclose(m.mixed(bank,cv,origin,np.ones(2),sets)[0,:3],2.)
    np.testing.assert_allclose(m.mixed(bank,cv,origin+500.,np.ones(2),sets),
                               m.mixed(bank,cv,origin,np.ones(2),sets))


def test_whole_set_difference_uses_both_action_features():
    x = np.array([[[1.],[2.]],[[3.],[5.]],[[2.],[9.]]])
    cv = np.zeros((3,12,2))
    bank = np.zeros((3,2,12,2))
    y = np.zeros((2,3,2))
    y[0,0,0]=y[0,1,1]=y[1,0,1]=y[1,2,0]=1.
    d = m.demonstration_differences([np.arange(3)],x,bank,cv,np.zeros((3,2)),np.ones(3),y)
    np.testing.assert_allclose(d,[[.5,-1.5,0.,0.,0.,0.]])
