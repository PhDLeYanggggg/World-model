import numpy as np
import torch

from scripts.audit_m3w_opendd_tangent_budget import reallocate
from src.world_model.m3w_cv_tangent import CVTangent, directional_derivative, losses


def test_one_sided_gradient_respects_cusp_and_observation_mask():
    residual = torch.tensor([[[2., 0.], [0., 0.], [3., 0.]]], dtype=torch.float64)
    step = torch.tensor([[[-1., 1.], [0., 2.], [-9., 0.]]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    slope = directional_derivative(residual, step, mask)
    assert slope.item() == .5
    eps = 1e-6
    numeric = ((residual+eps*step).norm(dim=-1)-residual.norm(dim=-1))[mask].mean()/eps
    torch.testing.assert_close(numeric, slope[0], rtol=1e-5, atol=1e-6)


def test_convex_secant_exceeds_tangent_along_the_same_ray():
    g = torch.Generator().manual_seed(88)
    r = torch.randn(16, 12, 2, generator=g, dtype=torch.float64)
    d = torch.randn(16, 12, 2, generator=g, dtype=torch.float64)
    mask = torch.ones(16, 12, dtype=torch.bool)
    first = directional_derivative(r, d, mask)
    secant = ((r+d).norm(dim=-1)-r.norm(dim=-1)).mean(1)
    assert (secant >= first-1e-10).all()


def test_shape_preserves_radius_and_exact_cv_for_zero_budget():
    net = CVTangent(3)
    x = torch.randn(2, 3)
    unit = torch.tensor([[[1., 0.]]*12, [[0., 1.]]*12])
    step, logit = net(x, unit, torch.tensor([.25, 0.]))
    torch.testing.assert_close(step.norm(dim=-1).amax(1), torch.tensor([.25, 0.]))
    cv = torch.zeros_like(step)
    valid = torch.tensor([[True]*12, [False]*12])
    y = torch.where(valid[..., None], torch.ones_like(step), torch.nan)
    value = losses(step, logit, cv, y, valid, torch.tensor([True, False]), True)
    value.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_budget_redistribution_cannot_move_across_queries_or_drop_support():
    radius = np.array([0., 1., 4., 1., 2.])
    prob = np.array([.2, .9, .4, .1, .9])
    out = reallocate(radius, prob, [0, 3, 5])
    np.testing.assert_allclose(np.add.reduceat(out, [0, 3]), [2.5, 1.5])
    assert (out > 0).tolist() == (radius > 0).tolist()
    assert out[1] > .5*radius[1]
