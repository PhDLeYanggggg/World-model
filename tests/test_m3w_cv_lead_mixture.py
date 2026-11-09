import numpy as np
import torch

from scripts import run_m3w_cv_lead_mixture as study
from scripts import audit_m3w_cv_minimum_edit_count as audit


def test_initial_shared_and_lead_proposals_are_identical_and_bounded():
    x = torch.randn(2,7)
    p = torch.randn(2,6,12,2)
    net = study.model.LeadMixture(7,6)
    a,b = [net(x,p,k) for k in (False,True)]
    torch.testing.assert_close(a,b)
    assert torch.all(b>=p.amin(1)-1e-6) and torch.all(b<=p.amax(1)+1e-6)


def test_lead_specific_weights_are_smooth_and_not_constant():
    net = study.model.LeadMixture(2,2)
    x = torch.zeros(1,2)
    p = torch.zeros(1,2,12,2)
    p[:,1,:,0] = 1.
    with torch.no_grad():
        net.head[-1].bias.copy_(torch.tensor([6.,0.,3.,3.,0.,6.]))
    lead = net(x,p,True)[0,:,0]
    shared = net(x,p,False)[0,:,0]
    torch.testing.assert_close(shared,torch.full_like(shared,.5))
    assert torch.all(torch.diff(lead)>0.) and 0.<lead[0]<.01<.99<lead[-1]<1.


def test_missing_future_contributes_path_debit_not_label_value():
    cv = torch.zeros(1,12,2)
    ols = torch.ones_like(cv)
    missing = torch.full_like(cv,float('nan'))
    valid = torch.zeros(1,12,dtype=torch.bool)
    scale = torch.ones(1)
    loss = lambda p:study.model.objective(p,cv,ols,missing,valid,scale,1.)
    assert torch.isfinite(loss(cv)) and loss(cv)<loss(ols)


def test_radius_matching_preserves_recipient_mask_and_equal_peak():
    c = np.zeros((2,12,2))
    p,q = c.copy(),c.copy()
    p[0,5,0],q[0,11,1] = 2.,3.
    mask = np.array([True,False])
    a,b = study.match_radius(c,p,q,mask)
    np.testing.assert_array_equal(study.peaks(a,c),[2.,0.])
    np.testing.assert_array_equal(study.peaks(a,c),study.peaks(b,c))


def test_perfect_edit_lower_bound_still_counts_untouched_ols4_advantage():
    cv = np.zeros((3,2,2))
    target = np.ones_like(cv)
    q = np.array([0,0,0])
    assert audit.minimum(cv,target,target,np.ones((3,2),bool),q,2) is None
    ols = .9*target
    assert audit.minimum(cv,ols,target,np.ones((3,2),bool),q,None)==3
