import torch

from src.world_model.m3w_hindcast_residual import emitted, objective


def test_initial_forecast_is_exact_anchor_and_gradients_are_finite():
    raw = torch.zeros(3, 6, requires_grad=True)
    cv = torch.zeros(3, 12, 2)
    anchor = torch.full_like(cv, .1)
    valid = torch.zeros(3, 12, dtype=torch.bool)
    valid[0] = True
    y = torch.where(valid[..., None], .05, torch.nan)
    pred = emitted(raw, anchor)
    assert torch.equal(pred, anchor)
    loss = objective(pred, cv, anchor, y, valid, torch.tensor([True, False, False]), 'easy_guard')
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_harm_penalty_cannot_cancel_against_easy_benefit_and_ignores_missing_targets():
    cv = torch.zeros(3, 12, 2)
    anchor = cv.clone()
    y = torch.ones_like(cv)
    pred = torch.stack([torch.full_like(cv[0], 3.), torch.ones_like(cv[0]), torch.ones_like(cv[0])])
    valid = torch.ones(3, 12, dtype=torch.bool)
    valid[2] = False
    easy = torch.tensor([True, True, False])
    observed = objective(pred, cv, anchor, y, valid, easy, 'observed')
    completion = objective(pred, cv, anchor, y, valid, easy, 'completion')
    safe = objective(pred, cv, anchor, y, valid, easy, 'easy_guard')
    sqrt_two = torch.tensor(2.).sqrt()
    torch.testing.assert_close(completion-observed, sqrt_two/3)
    torch.testing.assert_close(safe-completion, 4*sqrt_two/3)
    y[2] = torch.nan
    torch.testing.assert_close(objective(pred, cv, anchor, y, valid, easy, 'easy_guard'), safe)
