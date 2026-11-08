import torch

from src.world_model.m3w_pointwise_reference_forecast import pointwise_excess, loss


def test_stronger_reference_changes_observed_gradient_between_error_levels():
    y = torch.zeros(1, 12, 2)
    p = torch.ones_like(y, requires_grad=True)
    valid = torch.ones(1, 12, dtype=torch.bool)
    cv, ols, scale = y+2, y+.5, torch.ones(1)
    normal = loss(p, y, valid, cv, ols, scale, 3., treatment=False)[0]
    joint = loss(p, y, valid, cv, ols, scale, 3., treatment=True)[0]
    diff = torch.autograd.grad(joint-normal, p)[0]
    assert (diff > 0.).all()
    assert pointwise_excess(p, y, valid, ols, cv, scale).item() == 0.


def test_unknown_coordinates_have_no_effect_on_added_penalty():
    y = torch.ones(2, 12, 2)
    p = torch.zeros_like(y, requires_grad=True)
    valid = torch.zeros(2, 12, dtype=torch.bool); valid[0,0] = True
    cv, ols, scale = p.detach()+1.1, p.detach()+1.2, torch.ones(2)
    known = pointwise_excess(p, y, valid, cv, ols, scale)
    y[~valid] = torch.nan
    hidden = pointwise_excess(p, y, valid, cv, ols, scale)
    torch.testing.assert_close(hidden, known)
    assert hidden[1].item() == 0.
    assert torch.isfinite(torch.autograd.grad(hidden.sum(), p)[0]).all()
