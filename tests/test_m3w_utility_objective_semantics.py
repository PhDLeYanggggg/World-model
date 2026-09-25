"""Objective semantics, not evidence of empirical neural-policy improvement."""
import torch

from src.world_model.m3w_native_gain_harm import cost_loss


def test_asymmetric_harm_objective_can_reverse_positive_mean_utility():
    # Mutually exclusive gain/harm outcomes with positive mean net benefit.
    target = torch.tensor([[1.5, 0.], [0., 1.]])
    conditional_mean = torch.tensor([[.75, .5], [.75, .5]])
    conservative_harm = torch.tensor([[.75, .8], [.75, .8]])
    assert cost_loss(conditional_mean, target, 'mse') < cost_loss(conservative_harm, target, 'mse')
    assert cost_loss(conservative_harm, target, 'underharm4') < cost_loss(conditional_mean, target, 'underharm4')
    assert (conditional_mean[:, 0]-conditional_mean[:, 1] > 0).all()
    assert (conservative_harm[:, 0]-conservative_harm[:, 1] < 0).all()


def test_asymmetric_harm_stationary_prediction_is_not_mean():
    target = torch.tensor([[1.5, 0.], [0., 1.]])
    harm = torch.tensor(.8, requires_grad=True)
    prediction = torch.stack((torch.full((2,), .75), harm.repeat(2)), 1)
    cost_loss(prediction, target, 'underharm4').backward()
    assert torch.isclose(harm.grad, torch.tensor(0.), atol=1e-7)
    assert not torch.isclose(harm.detach(), target[:, 1].mean())
