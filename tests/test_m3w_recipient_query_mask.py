from itertools import combinations

import numpy as np
import pytest
import torch

from src.world_model import m3w_recipient_query_mask as model


def test_allocate_matches_brute_force_on_same_count_and_spend_range():
    cost = np.array([1., 2., 1.95, .95, 0., 3.1, 2.01])
    score = np.array([0., .2, 1., .9, 100., 99., 98.])
    old = np.array([True, True, False, False, False, False, False])
    selected, status = model.allocate(score, cost, old)
    legal = [ids for ids in combinations(np.flatnonzero(cost > 0.), 2)
             if .95*cost[old].sum() <= cost[list(ids)].sum() <= cost[old].sum()]
    assert status == 'optimal'
    assert selected.sum() == old.sum()
    assert score[selected].sum() == max(score[list(x)].sum() for x in legal)


def test_one_and_zero_quota_and_invalid_anchor():
    x = np.array([.1, .9, 1.])
    c = np.array([1., .95, .94])
    old = np.array([True, False, False])
    np.testing.assert_array_equal(model.allocate(x, c, old)[0], [False, True, False])
    assert model.allocate(x, c, ~np.ones(3, dtype=bool))[1] == 'zero_quota'
    with pytest.raises(ValueError, match='Invalid'):
        model.allocate(x, c-1., old)


def test_pooling_is_permutation_invariant_and_empty_safe():
    x = np.array([[1., 2.], [3., 1.]], dtype=np.float32)
    np.testing.assert_array_equal(model.describe(x), model.describe(x[::-1]))
    np.testing.assert_array_equal(model.describe(x[:0]), np.zeros(7))


def test_propose_does_not_need_label_fields_and_reuses_nominated_actions():
    from scripts import run_m3w_opendd_recipient_query_mask as run

    h = np.arange(-7., 1.)[None, :, None]*np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]]])
    cv = np.zeros((6, 12, 2))
    f = dict(counts=[3, 3], history=np.tile(h, (2, 1, 1)), offsets=np.zeros((6, 2)), x=np.ones((6, 4)))
    edited = np.zeros((2, 2, 12, 2))
    edited[..., 0] = .5
    p = dict(cv=cv, bounds=[[0, 3], [3, 6]], additive_rows=np.array([[0, 1], [3, 4]]), additive_paths=edited)

    class Local(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[.3, -2.], [.1, 2.]])

    class Query(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([-.3, .7])

    value, status = run.propose(p, f, Local(), Query(), np.array([0, -1]))
    assert value['actions'].tolist() == [0, 0]
    assert value['query_mask'].tolist() == [False, True]
    assert status['query_mask'] == 'optimal'
