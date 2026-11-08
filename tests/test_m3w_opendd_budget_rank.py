import numpy as np
import torch

from src.world_model import m3w_opendd_budget_rank as m


def test_conservative_guard_uses_anchor_and_disallows_predicted_harm():
    scores = np.array([[.1, .1, 0., 0., 0., .6],
                       [.5, .1, .1, 0., 0., .8],
                       [.3, .1, 0., 0., 0., .7],
                       [.7, .8, 0., 0., 0., 1.]])
    assert m.choose(scores) == 1
    assert m.choose(scores, guarded=True) == 2
    assert m.choose(scores[[0, 3]], guarded=True) == 0


def test_no_reference_never_becomes_feasible():
    g, z = np.ones(2), -np.ones(3)
    assert m.risk_label(g, z, 1., 1., 1., 0.)
    assert not m.risk_label(g, z, 0., 1., 1., 0.)
    assert not m.risk_label(g, z, 1., 1., 0., 0.)
    assert not m.risk_label(g, z, 1., 1., 1., -2.)


def test_unresolved_gain_interval_adds_no_ranking_gradient():
    net = m.network(2)
    x = torch.zeros((2, 2))
    target = torch.tensor([[.2, .3, 0., 0., 0., 0.], [-.1, .1, 0., 0., 0., 0.]])
    torch.testing.assert_close(m.loss(net, x, target, True), m.loss(net, x, target, False))
