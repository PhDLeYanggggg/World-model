import numpy as np
import torch

from src.world_model import m3w_moving_observability as m


def test_missingness_debit_uses_exact_radius_and_zero_known_weight():
    a = np.array([[[2., -.5, -.25], [3., -.2, -.1]]])
    d = np.array([[4., 5.]])
    np.testing.assert_array_equal(m.costs(a, np.zeros((1, 1)), d), a)
    np.testing.assert_array_equal(m.costs(a, np.ones((1, 1)), d), [[[-4., 4., 4.], [-5., 5., 5.]]])


def test_observability_context_is_query_local_and_translation_invariant():
    h = np.random.default_rng(8).normal(size=(4, 8, 2))
    groups = (np.arange(2), np.arange(2, 4))
    a = m.observability_features(h, groups)
    h[2:] = h[2:]*10.
    np.testing.assert_allclose(a[:2], m.observability_features(h+5., groups)[:2])


def test_conditional_head_loss_is_finite_when_all_targets_are_missing():
    net = m.Head(5)
    y = torch.zeros((3, 4))
    y[:, 3] = 1.
    loss = m.loss(net, torch.zeros((3, 5+m.MOTION+m.GEOMETRY)), y)
    assert torch.isfinite(loss)
    loss.backward()
