import numpy as np
import torch

from src.world_model import m3w_opendd_reserve as m


def packet(rows, easy, gain, joint):
    return tuple(map(np.array, (rows, easy, gain, joint)))


def test_single_reserve_uses_prefix_and_never_forces_at_end():
    seq = [packet([1], [.7], [.4], [.1]),
           packet([3, 4], [.8, .4], [.6, .3], [.3, .5]),
           packet([8], [.9], [.9], [.9])]
    assert m.first(iter(seq[:1]), "joint") is None
    assert m.first(iter(seq[:2]), "joint") == (1, 4)
    assert m.first(iter(seq), "joint") == (1, 4)
    assert m.first(iter(seq), "easy") == (0, 1)
    assert m.pick(*seq[1], "easy", force=True) == 3


def test_stop_requires_positive_gain_and_probable_support():
    assert m.pick(*packet([2, 4], [.9, .2], [-.1, .6], [.9, .1]), "joint") is None
    assert m.first(iter([packet([], [], [], [])]), "joint") is None
    head = m.network(3)
    loss = m.loss(head, torch.zeros(2, 3), torch.tensor([[.1, 0., 1.], [-.2, 0., 0.]]))
    assert torch.isfinite(loss)
    loss.backward()


def test_factorized_eligibility_does_not_force_a_last_query():
    from scripts.audit_m3w_opendd_factorized_reserve import factorized

    seq = [tuple(map(np.array, ([1, 2], [.9, .1], [-.1, .9]))),
           tuple(map(np.array, ([3, 4], [.8, .7], [.1, .3])))]
    assert factorized(iter(seq[:1])) is None
    assert factorized(iter(seq)) == (1, 4)
