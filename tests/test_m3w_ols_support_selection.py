import numpy as np
import torch

from src.world_model.m3w_ols_support_selection import (choose, design, GainSupport, neighbor_windows,
                                                       objective, risk_score)


def test_missing_label_probability_subtracts_known_analytic_debit():
    cost = np.array([[[.2,.03],[.4,.01]]])
    debit = np.array([[.1,.6]])
    np.testing.assert_allclose(risk_score(cost,np.ones(1),debit,factorized=True),-2*debit)
    np.testing.assert_allclose(risk_score(cost,np.zeros(1),debit,factorized=True),[[.17,.39]])


def test_factorized_cost_has_no_gradient_for_missing_rows():
    cost = torch.zeros(2,3,2,requires_grad=True)
    logit = torch.zeros(2,requires_grad=True)
    t = torch.ones_like(cost)
    f,_,_ = objective(cost,logit,t,torch.tensor([False,True]),factorized=True)
    f.backward()
    assert (cost.grad[1] == 0).all() and (cost.grad[0] != 0).all()
    assert logit.grad[0] > 0 and logit.grad[1] < 0


def test_all_arms_share_candidate_support_but_free_arm_may_abstain():
    groups = [np.arange(3),np.array([3])]
    scores = np.array([[.1,-1],[-2,-3],[-3,-2],[1,1]])
    legal = np.ones_like(scores,dtype=bool)
    np.testing.assert_array_equal(choose(scores,legal,groups),[0,-1,-1,-1])
    np.testing.assert_array_equal(choose(-scores,legal,groups,counts=[1,0]),[-1,1,-1,-1])


def test_neighbor_translation_does_not_change_support_features():
    h = np.stack([np.column_stack((np.arange(8.)+q,np.zeros(8))) for q in (1.,4.,8.)])
    g = [np.arange(3)]
    f,n,*_ = design(h,*neighbor_windows(h,g))
    e,b,*_ = design(h+1000,*neighbor_windows(h+1000,g))
    np.testing.assert_allclose(f,e)
    np.testing.assert_allclose(n,b)
    net = GainSupport(f.shape[-1])
    c,s = net(torch.from_numpy(f),torch.from_numpy(n),ego_support=False)
    assert c.shape == (3,12,2) and s.shape == (3,)
