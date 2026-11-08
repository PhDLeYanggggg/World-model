import numpy as np
import torch

from src.world_model import m3w_opendd_slate_rank as m
from src.world_model.m3w_opendd_slate_queue import Packet


def test_shared_edited_rows_do_not_add_missing_step_debit():
    a, b = np.array([0, 1]), np.array([1, 2])
    gain, missing = np.array([4., -50., 1.]), np.array([1., 100., .5])
    assert m.matched_advantage(a, b, gain, missing) == (3., 1.5)
    assert m.matched_advantage(b, a, gain, missing) == (-3., 1.5)
    assert m.matched_advantage(a, a, gain, missing) == (0., 0.)


def test_same_count_examples_keep_ambiguity_and_harm_separate():
    sets = tuple(map(np.array, ([], [0], [1], [0, 1])))
    p = Packet(sets, np.eye(4), np.array([0, 1, 3]), np.array([0, 1, 1, 2]), 1, 2.)
    risk = np.array([[0., 2., 0.], [0., 0., 2.]])
    [(x, z)] = m.examples(p, 2, np.array([1., 0.]), np.array([1.5, 0.]), risk, np.array([2., 0.]))
    np.testing.assert_array_equal(x, p.features[[1, 2]])
    np.testing.assert_allclose(z, [0., 0., 1., -1., 1., 0.])
    for scalar in (True, False):
        net = m.network(x.shape[-1], scalar)
        loss = m.loss(net, torch.tensor(x[None], dtype=torch.float32), torch.tensor(z[None], dtype=torch.float32), scalar)
        assert torch.isfinite(loss)
        loss.backward()


def test_guard_blocks_gain_with_any_predicted_harm_increase():
    z = np.array([[.2, 1., 2., 4.], [.4, 0., 2., 4.], [.9, 0., 3., 4.]])
    assert m.decision(z, 0) == 2
    assert m.decision(z, 0, True) == 1
    assert m.decision(z[[0, 2]], 0, True) == 0
