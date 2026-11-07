import numpy as np
import torch

from src.world_model.m3w_common_motion import batch_loss, describe, make_net, shifted


def test_shift_preserves_all_relative_vectors_and_uses_convex_weights():
    rng = np.random.default_rng(8)
    h = rng.normal(size=(4, 8, 2))
    paths = rng.normal(size=(3, 4, 12, 2))
    x, offset = describe(h, paths)
    assert np.isfinite(x).all()
    moved = shifted(paths[0], offset, [.2, .7, .1])
    np.testing.assert_allclose(moved.mean(0), np.einsum('a,atd->td', [.2, .7, .1], paths.mean(1)))
    np.testing.assert_allclose(moved[:, None]-moved, paths[0, :, None]-paths[0], atol=1e-12)
    np.testing.assert_array_equal(shifted(paths[0], offset, [1., 0., 0.]), paths[0])


def test_unobserved_query_rows_charge_path_disagreement_not_fabricated_truth():
    net = make_net(2)
    z = torch.zeros((2, 2))
    offset = torch.zeros((2, 3, 12, 2))
    offset[1, :, :, 0] = 1.
    ols = torch.zeros((2, 12, 2))
    y = torch.full_like(ols, 300.)
    valid = torch.zeros((2, 12), dtype=torch.bool)
    loss = batch_loss(net, z, offset, ols, y, valid, torch.tensor([0, 1]))
    torch.testing.assert_close(loss, torch.tensor(.5))
