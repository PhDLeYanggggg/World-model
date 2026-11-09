import numpy as np
import torch

from src.world_model.m3w_exact_slot_costs import CostHead, choose, ranks, training_examples


def paths():
    h = np.zeros((3, 8, 2))
    h[:, :, 0] = np.arange(8.)
    h[:, :, 1] = np.arange(3.)[:, None]
    cv = np.zeros((3, 12, 2))
    cv[:, :, 0] = np.arange(1., 13.)
    ols = .9*cv
    return h, cv, ols, np.stack((ols, 1.1*cv), axis=1)


def test_joint_only_changes_how_the_same_costs_are_aggregated():
    value = np.array([[2., 0., 1., 0., 0.], [0., 2., -1., 0., 0.]])
    a, j = ranks(value, np.array([[0, -1], [1, -1], [0, 1]]))
    np.testing.assert_array_equal(a, [-1., 0., -1.])
    np.testing.assert_array_equal(j, [-1., 0., 2.])


def test_targets_change_but_inputs_are_identical_when_future_changes():
    h, cv, ols, bank = paths()
    v = np.ones(cv.shape[:2], bool)
    a, y = training_examples(h, cv, ols, bank, .8*cv, v, easy_cut=1.)
    b, z = training_examples(h, cv, ols, bank, 1.3*cv, v, easy_cut=1.)
    torch.testing.assert_close(a, b, rtol=0., atol=0.)
    assert not torch.equal(y, z)
    assert y.shape[1] == 5


def test_decoder_preserves_slots_and_agrees_on_single_edits():
    h, cv, ols, bank = paths()
    net = CostHead().eval()
    a, j = choose(net, h, cv, ols, bank, [.5])
    np.testing.assert_array_equal(a, j)
    for p in choose(net, h, cv, ols, bank, [.5, .8]):
        peaks = np.sort(np.linalg.norm(p-cv, axis=-1).max(1))
        np.testing.assert_allclose(peaks, [0., .5, .8])
