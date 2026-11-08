import numpy as np
import torch

from src.world_model import m3w_cv_query_rank as m


def test_pairwise_regret_respects_order_and_ties():
    y = torch.tensor([[.3, .1], [.2, .2]])
    p = torch.tensor([[.2, .1], [.4, .7]])
    assert m.pair_loss(p, y, True) < m.pair_loss(p.flip(1), y, True)
    assert m.pair_loss(p[1:], y[1:], True) == 0
    assert m.pair_loss(y, y, False) == 0


def test_pairs_stay_in_query_and_exclude_non_actions():
    query = [np.arange(4), np.arange(4, 8)]
    legal = np.array([1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    sites = np.repeat(["a", "b"], 4)
    ij, source = m.pairs(query, legal, sites, 17)
    assert (ij[:, 0] != ij[:, 1]).all() and legal[ij].all()
    assert (sites[ij] == source[:, None]).all()
    np.testing.assert_array_equal(ij, m.pairs(query, legal, sites, 17)[0])


def test_constant_velocity_has_zero_prefix_preference():
    h = np.tile(np.arange(8.)[None, :, None], (3, 1, 2))
    np.testing.assert_allclose(m.prefix_control(h), 0.)
