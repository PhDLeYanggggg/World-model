import numpy as np
import torch

from src.world_model import m3w_av2_projected_reference as model


def test_projection_preserves_radius_and_handles_zero_direction():
    delta = torch.tensor([[[3., 4.], [0., 0.], [1., 1.]]], dtype=torch.float64)
    direction = torch.tensor([[[0., 2.], [-3., 5.], [0., 0.]]], dtype=torch.float64)
    p = model.project(direction, delta)
    torch.testing.assert_close(p, torch.tensor([[[0., 5.], [0., 0.], [1., 1.]]], dtype=p.dtype))
    torch.testing.assert_close(torch.linalg.vector_norm(p, dim=-1), torch.linalg.vector_norm(delta, dim=-1))


def test_geometry_features_use_radius_but_not_old_direction():
    h = np.array([[.1, .3], [-.4, .3]])[:, None]*np.arange(8)[None, :, None]
    c = np.diff(h[:, -2:], axis=1)*np.arange(1, 13)[None, :, None]
    d = dict(h=h, cv=c, ols4=c*.95)
    p, q = c.copy(), c.copy()
    p[:, :, 0] += .2
    q[:, :, 1] -= .2
    np.testing.assert_allclose(model.inputs(d, p, [0, 1])[0], model.inputs(d, q, [0, 1])[0], atol=1e-8)


def test_partial_future_cannot_change_loss_via_masked_coordinates():
    cv = torch.zeros((1, 2, 12, 2))
    r = torch.full_like(cv, .15, requires_grad=True)
    valid = torch.ones((1, 2, 12), dtype=torch.bool)
    valid[:, 0, -4:] = False
    easy = torch.tensor([[True, False]])
    y = torch.full_like(cv, .8)
    masked_y = torch.where(valid[..., None], y, 999.)
    args = (r, cv, cv, torch.ones_like(cv)*.2, torch.ones(1, 2))
    context = torch.zeros((1, 2))
    a = torch.stack(model.objectives(*args, y, valid, easy, context))
    b = torch.stack(model.objectives(*args, masked_y, valid, easy, context))
    torch.testing.assert_close(a, b)
    a.sum().backward()
    assert torch.isfinite(r.grad).all()
