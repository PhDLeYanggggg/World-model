import numpy as np
import torch

from src.world_model import m3w_matched_bridge_swaps as m


def test_choices_preserve_count_and_exclude_two_arms_for_same_agent():
    legal = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool)
    assert m.choices(legal, 1).tolist() == [[0, -1], [2, -1], [4, -1], [6, -1]]
    pair = m.choices(legal, 2)
    assert len(pair) == 5
    assert (pair[:, 0]//3 != pair[:, 1]//3).all()
    assert len(m.choices(legal[:2], 2)) == 0
    assert not m.gather(np.ones((9, 2)), m.choices(legal, 1))[:, 1].any()


def test_joint_value_is_permutation_invariant_and_masks_singleton_padding():
    x = torch.randn(3, 2, m.WIDTH)
    net = m.SwapValue(True)
    pair = torch.ones(3)
    q = torch.ones(3)*2
    torch.testing.assert_close(net(x, pair, q), net(x.flip(1), pair, q))
    absent = torch.zeros(3)
    other = x.clone()
    other[:, 1] += 50
    torch.testing.assert_close(net(x, absent, q), net(other, absent, q))


def test_choice_target_respects_easy_harm_and_uses_ols_offset():
    part = np.array([[[2., 3., 0., 1., 1.]], [[1., 1., 0., 0., 0.]]])
    a = np.array([[0, -1], [1, -1]])
    assert m.utility(part, a, 0., 1.).argmax() == 1
    np.testing.assert_allclose(m.utility(part, a, -10., 1.), [-10.975, -9.])


def test_tied_targets_have_no_gradient_at_a_tied_score():
    p = torch.zeros((2, 2), requires_grad=True)
    loss = m.objective(p, torch.ones_like(p), True)
    loss.backward()
    torch.testing.assert_close(p.grad, torch.zeros_like(p))
