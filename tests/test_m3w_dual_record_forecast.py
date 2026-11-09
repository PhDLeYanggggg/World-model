import torch

from src.world_model.m3w_dual_record_forecast import components, objective


def test_missing_step_charges_each_reference_and_positive_harm_is_not_netted():
    cv = torch.zeros((2,2,2))
    ols = cv.clone()
    y = torch.tensor([[[1.,0.],[0.,0.]],[[1.,0.],[1.,0.]]])
    mask = torch.tensor([[True,False],[True,True]])
    pred = torch.tensor([[[1.,0.],[3.,0.]],[[0.,0.],[0.,0.]]],requires_grad=True)
    w = torch.ones(2)
    gains,excess,_ = components(pred,cv,ols,y,mask,w,2.)
    torch.testing.assert_close(gains,torch.tensor([-2./3.,-2./3.]))
    assert (excess < 0.).all()  # a partial row is not a wholly unknown risk case
    objective(pred,cv,ols,y,mask,w,2.,True).backward()
    assert torch.isfinite(pred.grad).all()


def test_fallback_does_not_manufacture_easy_support_and_wholly_missing_edits_cost_risk():
    cv = torch.zeros((2,2,2))
    y = torch.ones_like(cv)
    mask = torch.tensor([[True,True],[False,False]])
    w = torch.ones(2)
    _,_,support = components(cv,cv,cv,y,mask,w,2.)
    assert support > 0
    changed = cv.clone()
    changed[1,:,0] = 1.
    _,excess,_ = components(changed,cv,cv,y,mask,w,2.)
    assert (excess[:2] > 0.).all()
