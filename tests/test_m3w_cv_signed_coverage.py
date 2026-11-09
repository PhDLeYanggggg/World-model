import numpy as np
import torch

from src.world_model import m3w_cv_signed_coverage as study
from scripts.audit_m3w_cv_signed_failures import flags
from scripts.audit_m3w_cv_signed_matched_gain import matched_query


def test_conditional_decoder_allocates_missing_radius_to_both_budgets():
    logits = torch.tensor([[2.,-1.,0.,0.,0.]])
    radius = torch.tensor([3.])
    expected = torch.tensor([[2.*(2/3)+1.,-1/3+1.]])
    cost,p = study.decode_risk(logits,radius,True)
    torch.testing.assert_close(cost,expected)
    torch.testing.assert_close(p,torch.tensor([1/3]))
    direct,_ = study.decode_risk(logits,radius,False)
    torch.testing.assert_close(direct,logits[:,:2])


def test_conditional_loss_masks_non_easy_and_wholly_unknown_targets():
    x = torch.zeros(3,5,requires_grad=True)
    target = torch.ones(3,2)
    cat = torch.tensor([0,1,2])
    study.risk_loss(torch.nn.Identity(),x,target,cat,True).backward()
    assert (x.grad[:,0]<0.).tolist()==[True,False,True]
    assert (x.grad[:,1]<0.).tolist()==[True,False,False]


def test_query_swap_debits_missing_path_width_on_both_agents():
    cv = np.zeros((2,2,2,2))
    p = cv.copy()
    p[:,0,:,0] = [4.,1.]
    p[:,1,:,0] = [1.,1.]
    y = cv.copy()
    y[:,:,0,0] = 4.
    y[:,:,1] = np.nan
    v = np.broadcast_to([True,False],y.shape[:-1]).copy()
    p[1,1,1,0] = 2.
    scale = np.ones((2,2))
    np.testing.assert_array_equal(study.signed_swap(cv,p,y,v,scale),[.5,0.])
    np.testing.assert_array_equal(study.signed_swap(cv[:,::-1],p[:,::-1],y[:,::-1],v[:,::-1],scale),[-.5,0.])


def test_swap_matched_query_can_change_recipient_but_not_count():
    g = np.array([[1.],[2.],[3.]])
    r = np.array([[[-1.,-1.]],[[0.,0.]],[[0.,0.]]])
    support = np.array([[.5],[0.],[0.]])
    a,b = matched_query(g,g[::-1].copy(),r,support,np.ones_like(g,bool),2)
    np.testing.assert_array_equal(a,[0,-1,0])
    np.testing.assert_array_equal(b,[0,0,-1])


def test_known_only_diagnostic_does_not_treat_missing_support_as_pass():
    base = dict(known_all_risk=.01,known_easy_risk=.01,known_easy_net_degradation=.01,
                supported=False,unknown_selected=1)
    assert flags(base)['missing_debit_flips_known_pass']==1
    assert flags(dict(base,known_easy_risk=None))['known_only_pass']==0
