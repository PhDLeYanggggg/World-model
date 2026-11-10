import numpy as np
import torch

from src.world_model.m3w_vehicle_nominee_rank import WIDTH, budget_pair_loss, features


def test_features_change_when_nominee_changes():
    x = np.random.default_rng(17).normal(size=(2, 12, 50))
    a = features(x, 0, 1, -0.2, 5)
    b = features(x, 1, 4, -0.2, 5)
    assert a.shape == (WIDTH,)
    np.testing.assert_array_equal(a[:152], b[:152])
    assert not np.array_equal(a[152:-1], b[152:-1])


def test_budget_loss_has_only_within_source_pairs_and_hard_negative_gradient():
    z = torch.tensor([.8, .9, .1, .2, 1.5], requires_grad=True)
    labels = torch.tensor([1., 0., 0., 0., 0.])
    groups = torch.tensor([0, 0, 0, 0, 1])
    budget_pair_loss(z, labels, groups).backward()
    assert z.grad[0] < 0 < z.grad[1]
    torch.testing.assert_close(z.grad[2:], torch.zeros(3))
