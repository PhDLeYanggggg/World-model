import numpy as np
import pytest
import torch
from src.world_model import m3w_boundary_guard as api


def test_original_boundary_not_a_new_threshold_and_two_gradient_directions():
    p = torch.ones((2, 7, 5), dtype=torch.float64, requires_grad=True)
    y = torch.ones_like(p).detach()
    with torch.no_grad():
        p[:, :, 1] = .02; p[:, :, 4] = .02
        y[0, :, 1] = y[0, :, 4] = 0.
        y[1, :, 1] = y[1, :, 4] = .04
    legal = torch.ones((2, 7), dtype=torch.bool)
    loss, _ = api.boundary_loss(p, y, legal, torch.tensor([0, 1]), 2)
    loss.backward()
    assert (p.grad[0, :, [1, 4]] > 0).all()  # Lower falsely high harm.
    assert (p.grad[1, :, [1, 4]] < 0).all()  # Raise falsely low harm.
    assert p.grad[:, :, [0, 2, 3]].count_nonzero() == 0
    np.testing.assert_allclose(float(loss.detach()), np.log(2))


def test_unknown_actions_get_no_loss_or_gradient_and_no_nan():
    p = torch.zeros((3, 7, 5), dtype=torch.float64, requires_grad=True)
    y = torch.ones_like(p).detach()
    legal = torch.ones((3, 7), dtype=torch.bool); legal[1:] = False
    loss, info = api.boundary_loss(p, y, legal, torch.tensor([0, 1, 2]), 3)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    assert p.grad[1:].count_nonzero() == 0
    assert info['boundary_known_actions'] == 7


def test_objective_patch_is_restored_even_after_exception():
    before = api.parent.losses
    with pytest.raises(RuntimeError):
        with api.installed('boundary'):
            assert api.parent.losses is not before
            raise RuntimeError('interrupt')
    assert api.parent.losses is before
    with pytest.raises(ValueError):
        with api.installed('unregistered'):
            pass


def test_positive_native_rescaling_does_not_change_normalized_loss():
    p = torch.full((2, 7, 5), .3, dtype=torch.float64)
    y = p.clone(); y[0, :, 1] = y[0, :, 4] = 0
    scales = torch.tensor([2., 31.], dtype=torch.float64)
    mask = torch.ones((2, 7), dtype=torch.bool); segments = torch.tensor([0, 1])
    a, _ = api.boundary_loss(p, y, mask, segments, 2)
    b, _ = api.boundary_loss((p * scales[:, None, None]) / scales[:, None, None],
        (y * scales[:, None, None]) / scales[:, None, None], mask, segments, 2)
    torch.testing.assert_close(a, b)
