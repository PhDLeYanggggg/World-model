import numpy as np
import torch

from scripts.audit_m3w_cv_recipient_radius import radius_control
from scripts.build_m3w_cv_recipient_teachers import family
from src.world_model import m3w_cv_recipient_path as model


def paths(x):
    z = np.zeros((len(x),12,2))
    z[:,:,0] = np.asarray(x)[:,None]
    return z


def test_soft_teacher_counts_only_verified_analytic_subsets():
    c,o,y = paths([1]*4),paths([.95]*4),paths([0]*4)
    b = paths([.5]*4)[:,None]
    v,q = np.ones((4,12),bool),np.array([0,0,1,1])
    old = np.column_stack((np.arange(4),np.zeros(4,int)))
    soft,n,k = family(c,o,b,y,v,q,(2.,5.),old,old[:1],14)
    assert n==k and n>1
    assert ((soft>0)&(soft<1)).any()
    assert np.all((soft>=0)&(soft<=1))


def test_path_ranking_cannot_increase_recipient_displacement():
    cost = np.zeros((3,3,6))
    pair = cost.copy()
    cost[:,:,0:2] = np.array([1,3,2])[None,:,None]
    pair[:,:,0:2] = np.array([2,1,4])[None,:,None]
    legal = np.array([[1,1,1],[1,1,1],[0,0,0]],bool)
    radius = np.array([[1,2,4],[1,4,2],[0,0,0]])
    rank,a,b = model.nominations(cost,pair,legal,radius)
    np.testing.assert_array_equal(a[:2],[1,1])
    np.testing.assert_array_equal(b[:2],[0,2])
    assert not np.isfinite(rank[2])
    groups = [np.arange(3)]
    match = np.array([0,-1,-1])
    s = np.ones(3)
    np.testing.assert_array_equal(model.select(rank,a,groups,s,match=match)>=0,
                                  model.select(rank,b,groups,s,match=match)>=0)


def test_paired_loss_is_finite_when_all_actions_tie():
    x = torch.zeros((4,2,3))
    y = torch.zeros((4,2,5))
    net = torch.nn.Linear(3,6)
    loss = model.paired_loss(net,x,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(net.weight.grad).all()


def test_equal_radius_keeps_old_direction_and_new_peak():
    cv,old,new = paths([0,0]),paths([4,0]),paths([-1,0])
    z = radius_control(cv,old,new,np.array([True,False]))
    np.testing.assert_allclose(z,paths([1,0]))
