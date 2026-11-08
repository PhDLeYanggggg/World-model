import pytest
import torch

from src.world_model.m3w_completion_forecast import loss_terms, objective


def test_missing_step_is_debited_without_reading_the_label():
    p = torch.tensor([[[2., 0.], [4., 0.]]], requires_grad=True)
    y = torch.tensor([[[3., 0.], [float("nan"), 0.]]])
    cv, ols = torch.zeros_like(p), torch.ones_like(p) * torch.tensor([1., 0.])
    v, s = torch.tensor([[True, False]]), torch.tensor([2.])
    cost, a, b = loss_terms(p, y, v, s, cv, ols)
    assert [t.item() for t in (cost, a, b)] == pytest.approx([.25, .5, .5])
    cost.backward(retain_graph=True)
    assert p.grad[0, 1, 0] == 0
    p.grad.zero_()
    a.backward()
    assert p.grad[0, 1, 0] == .25
    assert torch.isfinite(p.grad).all()


def test_fully_observed_reference_term_changes_no_gradient():
    p = torch.tensor([[[2., 0.], [4., 0.]]], requires_grad=True)
    y = torch.tensor([[[3., 0.], [5., 0.]]])
    cv, ols = torch.zeros_like(p), torch.ones_like(p)
    v, s = torch.ones((1, 2), dtype=torch.bool), torch.ones(1)
    g = []
    for arm in ("masked", "cv_completion", "dual_completion"):
        l = objective(arm, p, y, v, s, cv, ols)
        g.append(torch.autograd.grad(l, p)[0])
    torch.testing.assert_close(g[0], g[1])
    torch.testing.assert_close(g[0], g[2])


def test_cv_identity_has_zero_cv_regret_even_without_labels():
    p = torch.ones((1, 12, 2))
    y = torch.full_like(p, float("nan"))
    v = torch.zeros((1, 12), dtype=torch.bool)
    s = torch.ones(1)
    assert objective("cv_completion", p, y, v, s, p, -p).item() == 0
    assert objective("dual_completion", p, y, v, s, p, -p).item() > 0


def test_completion_terms_are_tight_on_arbitrary_partial_paths():
    rng = torch.Generator().manual_seed(18)
    p, y, cv, ols = (torch.randn(9, 12, 2, generator=rng) for _ in range(4))
    v = torch.rand(9, 12, generator=rng) > .4
    s = torch.linspace(.5, 2., len(y))
    _, a, b = loss_terms(p, y, v, s, cv, ols)
    for ref, bound in ((cv, a), (ols, b)):
        def regret(completion):
            return (((p - completion).norm(dim=-1) -
                     (ref - completion).norm(dim=-1)) / s[:, None]).mean()

        assert regret(y) <= bound
        torch.testing.assert_close(regret(torch.where(v[..., None], y, ref)), bound)
