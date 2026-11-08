import numpy as np
import torch

from src.world_model import m3w_opendd_pass_transition as m


def test_transition_labels_are_mutually_exclusive():
    np.testing.assert_array_equal(m.targets(False, [True, False], [False, True]), [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(m.targets(True, [True, False], [False, True]), [[0, 0, 0], [0, 1, 1]])


def test_net_pass_score_accounts_for_destruction():
    p = np.array([[.7, .6, .0], [.5, .1, .2]])
    assert m.advantage(p).argmax() == 1
    assert m.advantage(p, rescue_only=True).argmax() == 0
    net = m.network(3)
    value = m.loss(net, torch.zeros((5, 3)), torch.zeros((5, 3)))
    assert torch.isfinite(value)
    value.backward()
