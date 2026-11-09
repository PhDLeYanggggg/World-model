import numpy as np
import torch

from src.evaluation.m3w_cv_anchor_budget import risk_excess
from src.world_model import m3w_cv_risk_components as model


def test_moments_reconstruct_the_original_risk_for_partial_and_unknown():
    cv = np.zeros((3, 12, 2))
    y = cv + np.array([1., 2., 3.])[:, None, None]
    v = np.ones((3, 12), bool)
    v[0, 1:] = False
    v[1] = False
    y[~v] = np.nan
    p = cv.copy()
    p[0] -= .5
    p[1, -1, 0] = 4.
    p[2] -= .5
    s = np.array([1., 2., 3.])
    yhat, cat, radius = model.targets(cv, p, y, v, s, 2.)
    np.testing.assert_array_equal(cat, [1, 0, 2])
    r = model.risk(yhat) * s[:, None]
    np.testing.assert_allclose(r[:, :2], risk_excess(cv, p[:, None], y, v, 2.)[:, 0], atol=1e-7)
    np.testing.assert_allclose(yhat[1], [radius[1], radius[1], 0., 0., 0.])


def test_support_branch_does_not_depend_on_proposed_action():
    torch.manual_seed(5)
    net = model.Head(7, 12)
    x = torch.randn(4, 12)
    z = x.clone()
    z[:, 7:] += 2.
    a, h1 = model.decode(*net(x), torch.ones(4), True)
    b, h2 = model.decode(*net(z), torch.ones(4), True)
    torch.testing.assert_close(h1, h2)
    torch.testing.assert_close(a[:, 3:], b[:, 3:])
    assert not torch.equal(a[:, :3], b[:, :3])


def test_unknown_component_is_exact_radius_without_reference_credit():
    hist = torch.tensor([[40., -40., -40., 1., 1.]])
    action = torch.randn(1, 3)
    r = torch.tensor([3.])
    m, _ = model.decode(hist, action, r, True)
    torch.testing.assert_close(m[0], torch.tensor([3., 3., 0., 0., 0.]))


def test_coherent_moments_and_finite_shared_loss():
    torch.manual_seed(7)
    x = torch.randn(8, 12)
    net = model.Head(7, 12)
    cat = torch.arange(8) % 3
    for arm in (False, True):
        m, _ = model.decode(*net(x), torch.ones(8), arm)
        assert (m[:, 0] >= m[:, 1]).all()
        assert (m[:, 3] >= m[:, 4]).all()
        z = model.objective(net, x, torch.zeros_like(m), cat, torch.ones(8), arm)
        z.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters())
