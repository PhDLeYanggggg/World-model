import itertools

import numpy as np

from src.world_model.m3w_quarter_allocation import allocate, targets


def test_exact_additive_rank_with_legal_actions_and_original_count():
    rng = np.random.default_rng(20)
    costs = rng.uniform(size=(6, 3, 5))
    legal = rng.uniform(size=(6, 3)) > .2
    direct = np.array([0, -1, 0, -1, -1, -1])
    legal[[0, 2], 0] = True
    q = np.arange(5)
    got = allocate([q, np.array([5])], direct, costs, legal, 1.)
    score = costs[..., 0]-costs[..., 1:3].sum(-1)+.02*costs[..., 3:].sum(-1)
    pick = np.flatnonzero(got >= 0)
    exhaustive = [score[i, a]+score[j, b] for i, j in itertools.combinations(q, 2)
                  for a, b in itertools.product(range(3), repeat=2) if legal[i, a] and legal[j, b]]
    np.testing.assert_allclose(score[pick, got[pick]].sum(), max(exhaustive))
    assert got[5] == -1
    np.testing.assert_array_equal(allocate([q, np.array([5])], direct,
                                          np.zeros_like(costs), legal, 1.), direct)


def test_quarter_targets_include_missing_radius_and_signed_easy_support():
    c = np.zeros((3, 12, 2))
    b = np.zeros((3, 1, 12, 2))
    b[:, 0, :, 0] = 4.
    y = c.copy()
    y[:, :, 0] = [np.full(12, .25), np.ones(12), np.zeros(12)]
    mask = np.ones((3, 12), bool)
    mask[-1] = False
    got = targets(c, b, y, mask, np.arange(3), np.zeros(3, int), .5)
    np.testing.assert_allclose(got, [[-.5, .5, .5, .25, .25], [1., 0., 0., 1., 0.],
                                    [-1., 1., 1., 0., 0.]])
