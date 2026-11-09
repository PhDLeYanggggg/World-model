import numpy as np
import torch

from src.world_model import m3w_full_scene_harm_transfer as model


def test_missing_actor_has_zero_reference_and_nonzero_debit():
    p = np.zeros((2, 5, 12, 2))
    p[:, 1:, :, 0] = 2.
    y = np.zeros((2, 12, 2))
    y[0, :, 0] = 1.
    v = np.ones((2, 12), bool)
    v[1] = False
    moments, ref = model.targets(p, y, v, np.ones(2), 1.1)
    np.testing.assert_array_equal(moments[1], np.tile([-2., 2., 2.], (4, 1)))
    np.testing.assert_array_equal(ref, [[1., 1.], [0., 0.]])
    np.testing.assert_array_equal(moments[0], 0.)


def test_scene_policy_and_gain_control_choose_same_count_not_same_recipients():
    m = np.array([[[10., 1., 1.]], [[6., 0., 0.]], [[5., 0., 0.]]])
    r = np.full((3, 2), 1.)
    p = model.choose(m, r, np.ones((3, 1), bool))
    assert (p[:, 2] != 0).sum() == (p[:, 3] != 0).sum() == 0
    m[0, 0, 1:] = [.08, .08]
    # Make the unsafe action rank just below the safe choices by excess.
    m[0, 0, 0] = 6.01
    p = model.choose(m, r, np.ones((3, 1), bool))
    np.testing.assert_array_equal(p[:, 2], [0, 1, 0])
    np.testing.assert_array_equal(p[:, 3], [1, 0, 0])


def test_candidate_predictions_cannot_change_reference_mass():
    net = model.HarmHead(3, 4)
    a = torch.zeros((2, 3))
    o = torch.zeros((2, 4, 4))
    o[1] = 1.
    m, r = net(a, o)
    assert m.shape == (2, 4, 3) and r.shape == (2, 2)
    torch.testing.assert_close(r[0], r[1])
    assert (m[..., 1:] >= 0).all() and (r >= 0).all()
