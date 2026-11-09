import numpy as np
import torch

from scripts.run_m3w_signed_pair_geometry import Matched
from src.world_model.m3w_signed_pair_geometry import WIDTH, directional, slope
from src.world_model.m3w_slate_imitation import Slate


def test_directional_features_ignore_global_origin_rotation_and_agent_order():
    rng = np.random.default_rng(27)
    h = rng.normal(size=(7, 2, 8, 2))
    ref, delta = (rng.normal(size=(7, 2, 12, 2)) for _ in range(2))
    scale = np.full(7, 2.)
    features = directional(h, ref, delta, scale)
    assert features.shape == (7, WIDTH)
    rotation = np.array([[.6, -.8], [.8, .6]])
    other = directional(h[:, ::-1] @ rotation + 42., ref[:, ::-1] @ rotation,
                        delta[:, ::-1] @ rotation, scale)
    np.testing.assert_allclose(features, other, atol=1e-7, rtol=1e-6)


def test_velocity_features_keep_direction_of_approach():
    h = np.zeros((2, 2, 8, 2))
    h[:, 0, :, 0] = 8. + np.arange(-7, 1)[None] * np.array([1., -1.])[:, None]
    ref = np.zeros((2, 2, 12, 2))
    features = directional(h, ref, ref, [1., 1.])
    np.testing.assert_allclose(slope(h, 8)[:, 0, 0], [1., -1.])
    np.testing.assert_allclose(features[:, 1:3], [[1., 1.], [-1., -1.]])
    np.testing.assert_allclose(features[:, 4:], 0.)


def test_same_width_same_weights_and_identical_initial_scores():
    net = Matched(Slate())
    for p, q in zip(net.heads['magnitude'].parameters(), net.heads['signed'].parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0., atol=0.)
    a, b = net.effects(torch.randn(8, 52), torch.randn(8, WIDTH))
    torch.testing.assert_close(a, b, rtol=0., atol=0.)
    assert not any(p.requires_grad for p in net.encoder.parameters())
