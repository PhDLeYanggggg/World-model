import torch

from src.world_model.m3w_dual_reference_forecast import causal_references, objective


def test_references_respect_current_origin_and_constant_velocity():
    h = torch.arange(8, dtype=torch.float32)[None, :, None].expand(2, 8, 2)
    cv, ols = causal_references(h)
    torch.testing.assert_close(cv, ols)
    torch.testing.assert_close(cv[:, -1], torch.full((2, 2), 12.))


def test_unknown_uses_each_own_reference_and_never_hidden_targets():
    p = torch.zeros(2, 2, 2, requires_grad=True)
    cv, ols = torch.zeros_like(p), torch.ones_like(p)
    y = torch.zeros_like(p)
    y[0, 1] = torch.nan
    y[1] = torch.nan
    mask = torch.tensor([[True, False], [False, False]])
    scale = torch.ones(2)
    base, g, penalty = objective(p, y, mask, cv, ols, scale, 2., dual=False)
    # Known gain relative to OLS cancels the equally large unknown debit.
    torch.testing.assert_close(g, torch.zeros(2))
    assert base.item() == penalty.item() == 0.
    y[0, 1] = 1e6
    y[1] = -1e6
    changed, _, _ = objective(p, y, mask, cv, ols, scale, 2., dual=True)
    torch.testing.assert_close(base, changed)
    changed.backward()
    assert torch.isfinite(p.grad).all()


def test_dual_objective_penalizes_disagreement_with_ols_even_at_cv():
    p = torch.zeros(1, 3, 2, requires_grad=True)
    cv, ols = torch.zeros_like(p), torch.ones_like(p)
    m = torch.zeros((1, 3), dtype=torch.bool)
    scale = torch.ones(1)
    base, _, _ = objective(p, cv, m, cv, ols, scale, 2., dual=False)
    dual, channels, _ = objective(p, cv, m, cv, ols, scale, 2., dual=True)
    assert base.item() == 0. and dual.item() > 0.
    assert channels[1] > channels[0]
    dual.backward()
    assert (p.grad < 0.).any()
