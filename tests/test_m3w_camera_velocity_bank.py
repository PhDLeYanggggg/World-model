import numpy as np
import pytest
import torch

from src.world_model.m3w_camera_velocity_bank import VelocityMixer, features, loss


def test_all_velocity_anchors_recover_constant_motion():
    v = np.array([[1., -2.], [0., 0.]])
    h = np.array([[3., 2.], [-7., 5.]])[:, None]+np.arange(8)[None, :, None]*v[:, None]
    x, slopes, scale = features(h)
    assert x.shape == (2, 16)
    assert np.isfinite(x).all()
    assert (scale > 0.).all()
    np.testing.assert_allclose(slopes, np.broadcast_to(v[:, None], slopes.shape))


def test_equivariance_and_convex_support():
    h = np.random.default_rng(17).normal(size=(8, 8, 2))
    x, v, scale = features(h)
    # Coordinatewise median slopes commute with this signed axis rotation.
    rot = np.array([[0., 1.], [-1., 0.]])
    z, u, other = features(3.*h@rot+7.)
    np.testing.assert_allclose(z, x, atol=1e-7)
    np.testing.assert_allclose(u, 3.*v@rot)
    np.testing.assert_allclose(other, 3.*scale)
    net = VelocityMixer()
    prediction, w = net(torch.from_numpy(x), torch.from_numpy(v.astype(np.float32)))
    torch.testing.assert_close(w.sum(1), torch.ones(8))
    assert (w >= 0.).all()
    assert (prediction.detach().numpy() >= v.min(1)-1e-6).all()
    assert (prediction.detach().numpy() <= v.max(1)+1e-6).all()


def test_guard_penalizes_harm_but_not_fully_masked_rows():
    cv = torch.tensor([[1., 0.], [0., 0.], [9., 0.]])
    v = torch.tensor([[2., 0.], [0., 0.], [9., 0.]], requires_grad=True)
    y = torch.tensor([[1.05, 0.], [3., 0.], [999., 999.]])[:, None]*torch.arange(1, 13)[None, :, None]
    valid = torch.ones((3, 12), dtype=torch.bool)
    valid[-1] = False
    guarded, ade, guard = loss(v, cv, y, valid, preserve=True)
    clean, _, inactive = loss(v, cv, y, valid, preserve=False)
    assert guarded > clean == ade
    assert guard > 0 and inactive == 0
    guarded.backward()
    assert torch.isfinite(v.grad).all()
    assert torch.all(v.grad[-1] == 0)


def test_future_positions_cannot_extend_the_feature_input():
    with pytest.raises(ValueError, match='eight finite past positions'):
        features(np.ones((2, 20, 2)))
