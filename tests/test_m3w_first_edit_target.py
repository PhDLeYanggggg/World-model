import numpy as np
import torch

from src.world_model import m3w_first_edit_target as m


def test_nested_first_edit_target_accounts_for_recording_offset_and_risk():
    gain = np.array([[1., 2.], [1., .9], [1., 2.], [1., 2.]])
    risk = np.array([[0., 0., 0.], [0., 0., 0.], [0., .1, 0.], [0., 0., 0.]])
    result = m.labels(gain, risk, np.array([True, True, True, False]), -1., 1.)
    assert result.tolist() == [[True, True, True], [True, True, False], [True, False, False], [False, False, False]]


def test_no_positive_batch_can_still_train():
    net = m.network(3)
    loss = m.loss(net, torch.randn(6, 3), torch.zeros(6, 8))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_factorized_and_direct_readouts_use_distinct_heads():
    z = np.zeros((2, 9))
    z[0, 8] = 1.
    z[1, 6] = 1.
    p = m.scores(z)
    assert p["direct"].argmax() == 0
    assert p["factorized"].argmax() == 1
