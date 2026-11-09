import numpy as np
import pytest
import torch

from src.world_model import m3w_av2_release_rank as model


def test_query_balanced_cross_scene_pairs_use_genuine_supported_negatives():
    labels = [
        np.array([[True, True], [False, False]]),
        np.array([[False, False], [False, True]]),
        np.array([[False, False]]),
    ]
    pools = model.label_pools(labels)
    draws = list(model.sample(pools, np.random.default_rng(17), 8, 4))
    for p, ns in draws:
        assert p == (0, 0)
        assert all(i != p[0] and not labels[i][j, 0] for i, j in ns)
        assert ns[0] == ns[2] == (1, 1)
    assert draws == list(model.sample(pools, np.random.default_rng(17), 8, 4))


def test_no_invalid_positive_or_within_scene_only_pair():
    with pytest.raises(ValueError, match="Invalid"):
        model.label_pools([np.array([[True, False]])])
    with pytest.raises(ValueError, match="No cross-scene"):
        list(model.sample(model.label_pools([np.array([[True, True], [False, True]])]),
                          np.random.default_rng(29), 1, 4))


def test_ranking_loss_penalizes_the_highest_wrong_score():
    pos = torch.tensor([0.], requires_grad=True)
    neg = torch.tensor([[2., -2.]], requires_grad=True)
    rank_loss, point, pair = model.loss(pos, neg, 1.)
    assert torch.allclose(model.loss(pos, neg, 0.)[0], point)
    assert torch.allclose(rank_loss, point+pair)
    pair.backward()
    assert pos.grad[0] < 0 < neg.grad[0, 0]
    assert neg.grad[0, 1] == 0
