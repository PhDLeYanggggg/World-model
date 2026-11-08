import numpy as np
import torch

from src.world_model import m3w_opendd_protected_gain as m


def test_support_path_is_exactly_frozen_and_gain_moves_only_after_support():
    a = np.zeros((12, 2))
    b = np.ones((12, 2))
    old = ((0, 10, a), (1, 20, b))
    moved = ((0, 12, a), (1, 20, b.copy()))
    changed_support = ((0, 10, a), (1, 20, a))
    assert not m.allowed(old, moved, support_seen=False)
    assert m.allowed(old, moved, support_seen=True)
    assert m.allowed(old, moved, support_seen=False, protected=False)
    assert m.allowed(old, ((0, 10, b), (1, 20, b)), support_seen=False)
    assert not m.allowed(old, changed_support, support_seen=True)


def test_reference_guard_removes_high_value_but_risky_candidate():
    z = np.array([[0., 0., 0., 0., 0., .5, .2],
                  [3., .2, -.1, -.1, -.1, .6, .3],
                  [2., .2, -.1, -.1, -.1, .6, .1]])
    legal = np.array([True, True, True])
    assert m.choose(z, legal) == 1
    assert m.choose(z, legal, guard=True) == 2
    legal[2] = False
    assert m.choose(z, legal, guard=True) == 0


def test_sparse_reference_loss_is_finite_and_trainable():
    n = m.network(3)
    y = torch.zeros((2, 7))
    y[1, 6] = 1.
    loss = m.loss(n, torch.zeros((2, 3)), y)
    loss.backward()
    assert torch.isfinite(loss)
