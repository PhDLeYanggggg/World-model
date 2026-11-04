import math
import pytest
import torch
from src.world_model.m3w_regression_deferral_loss import two_stage_logistic


def test_equation_three_weights_not_oracle_onehot():
    logits = torch.tensor([[.2, -.3, 1.1]], dtype=torch.float64)
    cost = torch.tensor([[1., 4., 2.]], dtype=torch.float64)
    out, use = two_stage_logistic(logits, cost, torch.ones_like(logits, dtype=torch.bool), torch.tensor([True]))
    expected = -(torch.tensor([[6., 3., 5.]])*torch.log_softmax(logits, 1)).sum()/math.log(2)
    assert torch.allclose(out[0], expected) and use.item()


def test_stationary_distribution_routes_to_minimum_expected_cost():
    # Synthetic finite-cost population: this checks the implemented formula only.
    cost = torch.tensor([[.2, 1., .5], [.4, .8, .6]], dtype=torch.float64)
    mean = cost.mean(0); weights = mean.sum()-mean
    logits = weights.log().detach().requires_grad_()
    loss, _ = two_stage_logistic(logits.expand(2, -1), cost,
        torch.ones_like(cost, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    loss.mean().backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-12)
    assert logits.argmax() == mean.argmin()


def test_unknown_and_illegal_costs_do_not_contribute():
    logits = torch.zeros(3, 3, requires_grad=True)
    cost = torch.tensor([[1., 3., float('nan')], [float('nan')]*3, [float('nan')]*3])
    legal = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])
    loss, use = two_stage_logistic(logits, cost, legal, torch.tensor([True, False, True]))
    assert torch.equal(use, torch.tensor([True, False, False]))
    assert torch.isfinite(loss).all() and torch.equal(loss[1:], torch.zeros(2))
    loss.sum().backward()
    assert torch.equal(logits.grad[1:], torch.zeros_like(logits.grad[1:])) and logits.grad[0, 2] == 0
    with pytest.raises(ValueError):
        two_stage_logistic(logits, torch.full_like(logits, -1), legal, torch.ones(3, dtype=torch.bool))
