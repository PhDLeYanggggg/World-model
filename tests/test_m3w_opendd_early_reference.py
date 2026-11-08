import numpy as np
import torch

from src.world_model import m3w_opendd_early_reference as m


def packet(rows, gains, easy, radius, logits, mass):
    return tuple(np.asarray(z) for z in (rows, gains, easy, radius, logits, mass))


def test_early_support_keeps_gain_and_matched_radius_capacity():
    seq = [packet([0, 1, 2], [-1]*3, [.2, .4, .1], [2., 1., 3.], [1., 2., 1.], [1., 0., 0.]),
           packet([3, 4], [1., 0.], [.1, .9], [1., 1.], [0., 0.], [0., 0.])]
    assert m.choose(iter(seq), .5, "original") == (3, 4, 4)
    assert m.choose(iter(seq), .5, "mass") == (3, 0, 0)
    assert m.choose(iter(seq), .5, "incidence") == (3, 1, 1)


def test_never_displaces_gain_or_adds_a_late_slot():
    seq = [packet([0], [1.], [.9], [1.], [5.], [3.]),
           packet([1], [-1.], [.4], [1.], [5.], [3.])]
    assert m.choose(iter(seq), .5, "mass") == (0, None, None)


def test_shadow_control_can_change_recipient_without_amplifying():
    seq = [packet([0, 1, 2], [1., 0., 0.], [.1, .2, .4], [3., 1., 2.], [4., 2., -1.], [2., 3., 0.])]
    assert m.choose(iter(seq), .5, "mass") == (0, 1, 2)


def test_loss_handles_zero_feasible_rows():
    net = m.network(3)
    loss = m.loss(net, torch.zeros((4, 3)), torch.zeros((4, 2)), True)
    assert torch.isfinite(loss)
    loss.backward()
