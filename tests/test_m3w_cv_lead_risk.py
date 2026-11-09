import numpy as np
import torch

from scripts import run_m3w_cv_lead_risk as run
from scripts.audit_m3w_cv_late_edit_taper import redistribute
from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model import m3w_cv_lead_risk as model


def test_exact_visibility_and_benefit_reconstruct_original_target():
    c = np.zeros((2,3,2))
    o = np.ones_like(c)
    p = c.copy()
    p[...,0] = [.5,2.,.9]
    y = np.full_like(c,.6)
    v = np.array([[True,False,False],[False,True,True]])
    y[~v] = np.nan
    s = np.array([2.,3.])
    benefit = model.observed_targets(c,p,y,v,s)
    flat,lead = model.score(c,o,p,benefit,v.astype(float),s)
    _,_,target,_ = candidate_terms(c,o,p[:,None],y,v)
    np.testing.assert_allclose(lead,target[:,0].mean(1)/3/s)
    assert not np.allclose(flat,lead)


def test_tail_loss_changes_only_underestimated_easy_component():
    pred = torch.zeros((3,5),requires_grad=True)
    target = torch.tensor([[.1,.1],[.1,-.1],[.1,.1]])
    cat = torch.tensor([0,0,2])
    model.tail_loss(torch.nn.Identity(),pred,target,cat).backward()
    torch.testing.assert_close(pred.grad[:,0],torch.full((3,),-.1/3))
    torch.testing.assert_close(pred.grad[:,1],torch.tensor([-.15,.05,0.]))


def test_matched_factorial_keeps_same_count_with_different_eligibility():
    r = np.array([[[-1.,-1.]],[[0.,0.]],[[0.,0.]]])
    g = np.array([[1.],[2.],[3.]])
    p = np.array([[.5],[0.],[0.]])
    legal = np.ones_like(g,bool)
    changed = legal.copy()
    changed[2] = False
    left,right = run.matched_query([(g,r,p,legal),(g,r,p,changed)],3)
    np.testing.assert_array_equal(left,[0,-1,0])
    np.testing.assert_array_equal(right,[0,0,-1])


def test_late_taper_preserves_recipient_and_total_displacement():
    cv = np.zeros((2,12,2))
    path = cv.copy()
    path[0,:,0] = np.arange(1,13)
    u,t = redistribute(cv,path)
    assert (u[0,:,0]>0.).all() and (t[0,8:]==0.).all()
    np.testing.assert_allclose(np.linalg.norm(u-cv,axis=-1).sum(1),
                              np.linalg.norm(t-cv,axis=-1).sum(1))
    np.testing.assert_array_equal(u[1],t[1])
