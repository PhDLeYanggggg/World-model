import numpy as np
import pytest
import torch

from src.world_model import m3w_local_ols_proposal as model


def test_past_description_is_rotation_and_translation_invariant_for_moving_agents():
    h = np.arange(8.)[None, :, None]*np.array([[[2., 3.]], [[-3., 1.]]])
    h[1] += [4., 10.]
    r = np.array([[0., -1.], [1., 0.]])
    a = model.describe(h, [[0, 1]])
    b = model.describe(h@r + [12., 60.], [[0, 1]])
    for i in (0, 1, 3):
        np.testing.assert_allclose(a[i], b[i], atol=1e-12)
    with pytest.raises(ValueError, match='partition'):
        model.describe(h, [[0], [0]])


def test_proposal_is_bounded_and_unknown_labels_do_not_affect_loss():
    net = model.make_net(2)
    x, rot, scale = torch.ones(3, 2), torch.eye(2).expand(3, -1, -1), torch.arange(1., 4.)
    assert model.shift(net, x, rot, scale).count_nonzero() == 0
    with torch.no_grad():
        net[-1].bias.fill_(10.)
    d = model.shift(net, x, rot, scale)
    assert (torch.linalg.vector_norm(d, dim=-1) <= .5*scale[:, None]*torch.arange(1., 13.)).all()
    valid = torch.zeros(3, 12, dtype=torch.bool)
    valid[0, 0] = True
    base = torch.zeros_like(d)
    a = model.loss(d, base, torch.where(valid[..., None], base, torch.nan), valid, scale)
    b = model.loss(d, base, base, valid, scale)
    torch.testing.assert_close(a, b)
    a.backward()
    assert torch.isfinite(net[-1].weight.grad).all()
