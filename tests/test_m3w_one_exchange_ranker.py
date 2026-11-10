from itertools import product

import numpy as np
import pytest
import torch

from scripts.audit_m3w_opendd_one_exchange_ranking import matched_chance
from src.evaluation.m3w_count_motion_capacity import respects_tokens, within_cap
from src.world_model.m3w_one_exchange_ranker import exchange_menu, exchanged, features, objective, rank


def test_exchange_menu_matches_exhaustive_legal_one_exchange():
    n = 11
    legal = np.tile([True, True, False], (n, 1))
    legal[-2:, 2] = True
    anchor = np.zeros(n, dtype=int)
    anchor[[4, 9]] = 1
    movement = np.tile([0., 1., 2.], (n, 1))
    movement[6, 1] = 1.1
    cap = 2.
    q = np.arange(n)
    expected = set()
    for values in product(*(np.flatnonzero(row) for row in legal)):
        p = np.asarray(values)
        if (p.sum() == anchor.sum() and np.maximum(p-anchor, 0).sum() == 1
                and within_cap(movement[q, p].sum(), cap) and respects_tokens(p, legal[:, 1])):
            expected.add(tuple(p))
    pairs = exchange_menu(anchor, legal, movement, cap)
    got = {tuple(exchanged(anchor, i, j)) for i, j in pairs}
    assert got == expected and len(pairs) == len(got) > 0
    x = np.ones((n, 4), dtype=np.float32)
    f, delta = features(x, legal, movement, anchor, cap, pairs)
    assert f.shape == (len(pairs), 4*x.shape[1]+9)
    assert (delta <= 0.).all()


def test_rank_can_abstain_or_require_positive_predicted_gain():
    scores = torch.tensor([[0., 0., 3., -.1], [0., 0., 2., .1], [0., 3., 0., .3]])
    assert rank(scores) == 1
    assert rank(scores, guard=False) == 0
    assert rank(scores[-1:]) is None
    assert rank(scores[:0]) is None
    assert rank(scores[-1:], guard=False, margin=-float('inf')) == 0


def test_joint_training_objective_has_finite_gradients():
    p = torch.zeros((3, 4), requires_grad=True)
    v = objective(p, torch.tensor([0, 1, 2]), torch.tensor([-.1, 0., .2]))
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(p.grad).all()
    assert p.grad[:, 3].abs().sum() > 0.


def test_random_control_acts_in_exactly_the_same_views():
    pass_ = np.array([1, 1, 0])
    gain = np.array([1., 2., -3.])
    r = matched_chance(pass_, gain, 1, 2)
    assert r['chosen_dual'] == 0 and r['destroys'] == 1 and r['exchanged'] == 1
    assert r['random_destroyed'] == pytest.approx(1./3.)
    assert r['random_gain_mass'] == 0.
    r = matched_chance(pass_, gain, 0, None)
    assert r['chosen_dual'] == r['random_dual'] == r['exchanged'] == 0
    assert r['random_rescued'] == r['random_destroyed'] == 0.
