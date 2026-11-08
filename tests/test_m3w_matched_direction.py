import numpy as np
import torch

from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model.m3w_matched_direction import DirectionHead, circle_oracle, project_context, rotate


def test_use_shared_ego_frame_and_preserve_real_neighbor_distance():
    xy = np.zeros((3, 8, 2))
    xy[..., 0] = np.arange(8)[None, :]
    xy[1, :, 1] = 2.
    xy[2, :, 1] = 6.
    g, _, _ = encode_scene(xy, np.ones((3, 8), bool), np.arange(3))
    h = g[:, :16].reshape(3, 8, 2)
    nh = g[:, 38:166].reshape(3, 8, 8, 2)[:, :4]
    mask = g[:, 230:294].reshape(3, 8, 8)[:, :4].astype(bool)
    assert np.all(h[:, -1] == 0.)
    d = np.broadcast_to([1., 0.], (3, 12, 2))
    features = project_context(h, nh, mask, d, np.ones(3))
    np.testing.assert_allclose(np.sinh(features[:, 17]), [2., 2., 4.])
    rotation = np.array([[0., -1.], [1., 0.]])
    np.testing.assert_allclose(project_context(h@rotation, nh@rotation, mask, d@rotation,
                                              np.ones(3)), features)


def test_unobserved_neighbor_values_do_not_enter_and_ego_mask_is_strict():
    h = np.zeros((1, 8, 2))
    nh = np.full((1, 4, 8, 2), np.nan)
    mask = np.zeros((1, 4, 8), bool)
    d = np.broadcast_to([1., 0.], (1, 12, 2))
    assert not np.any(project_context(h, nh, mask, d, np.ones(1)))
    net = DirectionHead(22, use_neighbors=False)
    with torch.no_grad():
        net.head[-1].weight.fill_(.3)
    x = torch.zeros(1, 22)
    edit = torch.randn(1, 12, 2)
    a = net(x, edit)[0]
    x[:, -8:] = 100.
    torch.testing.assert_close(net(x, edit)[0], a)


def test_rotation_and_hindsight_preserve_every_waypoint_radius():
    d = torch.tensor([[[1., 0.], [0., -3.], [0., 0.]]], dtype=torch.float64)
    torch.testing.assert_close(rotate(d, torch.tensor([[1., 2., 3.]])).norm(dim=2), d.norm(dim=2),
                               rtol=1e-7, atol=1e-7)
    y = np.array([[[0., -2.], [99., 0.], [2., 2.]]])
    oracle = circle_oracle(d.numpy(), y, np.array([[True, False, True]]))
    np.testing.assert_allclose(oracle, [[[0., -1.], [0., -3.], [0., 0.]]])
