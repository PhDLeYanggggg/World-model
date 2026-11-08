import numpy as np
import torch

from src.world_model import m3w_tumdot_joint_teacher as m


def test_local_teacher_and_guard_preserve_hindcast_when_unhelpful():
    y = np.array([[0., -1., 0.], [.2, 0., .1]])
    np.testing.assert_array_equal(m.local_teacher(y, 1), [1, 0])
    raw = np.array([[[.5, .1, 0., 3.], [0., 0., 0., 1.], [.2, -.1, 0., 2.]]])
    np.testing.assert_array_equal(m.choose(raw, 1, False), [0])
    np.testing.assert_array_equal(m.choose(raw, 1, True), [2])


def test_joint_preference_loss_has_gradients_with_matched_actions():
    net = m.network(5)
    x = torch.randn(8, 6, 5)
    target = torch.zeros(8, 6, 4)
    target[:, 1, 3] = 1.
    value = m.loss(net, x, target, 2)
    value.backward()
    assert torch.isfinite(value) and all(p.grad is not None for p in net.parameters())
