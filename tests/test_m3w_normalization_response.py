import pytest
import torch

from scripts.audit_m3w_normalization_response import gradient_summary


def test_measured_parameter_gradient_matches_direct_mean_and_internal_chain_rule():
    model = torch.nn.Linear(2, 24, bias=False)
    torch.nn.init.zeros_(model.weight)
    before = model.weight.detach().clone()
    prediction = model(torch.ones(2, 2)).reshape(2, 12, 2)
    target = torch.zeros_like(prediction); target[..., 0] = 3.
    radius = torch.full((2,), 2.)
    legacy, vector = gradient_summary(model, prediction, target, radius, 'primary_log')
    direct = torch.autograd.grad(torch.linalg.vector_norm(prediction-target, dim=-1).mean(-1).log1p().mean(), model.weight, retain_graph=True)[0]
    torch.testing.assert_close(vector, direct.flatten().double(), rtol=1e-6, atol=1e-8)
    internal, _ = gradient_summary(model, prediction, target, radius, 'internal_log')
    assert internal['mean_per_row_parameter_gradient_norm']/legacy['mean_per_row_parameter_gradient_norm'] == pytest.approx(4/5)
    assert legacy['cancellation_ratio'] == pytest.approx(1.)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def test_internal_loss_requires_identifiable_positive_radius():
    model = torch.nn.Linear(2, 24)
    prediction = model(torch.ones(1, 2)).reshape(1, 12, 2)
    with pytest.raises(ValueError, match='support'):
        gradient_summary(model, prediction, torch.ones_like(prediction), torch.zeros(1), 'internal_log')
