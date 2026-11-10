import numpy as np
import pytest
import torch

from src.world_model import m3w_recipient_transport as model


def example():
    h = np.arange(-7., 1.)[None, :, None] * np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., 0.]]])
    cv = np.zeros((4, 12, 2))
    anchor = cv.copy()
    anchor[0, :, 0] = np.arange(1., 13.)
    anchor[1, :, 1] = 2.
    return h, np.zeros((4, 2)), np.ones((4, 5)), cv, anchor


def test_transport_preserves_stepwise_norms_and_rotates_heading():
    h, offsets, design, cv, anchor = example()
    drop, add, new, x = model.proposals(h, offsets, design, cv, anchor, [0, 1])
    np.testing.assert_array_equal(drop, [0, 1])
    np.testing.assert_array_equal(add, [2, 2])
    np.testing.assert_allclose(new[:, :, 0], -anchor[drop].max(2))
    np.testing.assert_allclose(new[:, :, 1], 0.)
    assert x.shape == (2, 44)


def test_all_missing_swap_has_minus_one_normalized_gain():
    h, offsets, design, cv, anchor = example()
    drop, add, new, _ = model.proposals(h, offsets, design, cv, anchor, [0, 1])
    y, valid = cv.copy(), np.zeros(cv.shape[:2], dtype=bool)
    np.testing.assert_allclose(model.targets(cv, anchor, y, valid, drop, add, new), -1.)
    y[add[0]] = new[0]
    valid[:] = True
    assert model.targets(cv, anchor, y, valid, drop, add, new)[0] == 1.


def test_no_swap_for_empty_and_nonpositive_and_reject_invalid_recipients():
    assert model.decision(np.zeros(3)) == model.decision(np.empty(0)) == -1
    assert model.decision(np.array([.4, .4, .3])) == 0
    h, offsets, design, cv, anchor = example()
    with pytest.raises(ValueError, match='distinct'):
        model.proposals(h, offsets, design, cv, anchor, [0, 0])
    a, b, new, x = model.proposals(h[:2], offsets[:2], design[:2], cv[:2], anchor[:2], [0, 1])
    assert len(a) == len(b) == len(new) == len(x) == 0
    assert model.network(x.shape[1])(torch.from_numpy(x)).shape == (0,)


def test_robust_loss_concentrates_gradient_on_worse_donor():
    scores = torch.tensor([[.3, .3], [-.3, -.3]], requires_grad=True)
    target = torch.ones_like(scores)
    mean = model.objective(scores, target, False)
    robust = model.objective(scores, target, True)
    mean.backward(retain_graph=True)
    assert scores.grad[0].abs().mean() == scores.grad[1].abs().mean()
    scores.grad.zero_()
    robust.backward()
    assert robust > mean
    assert scores.grad[1].abs().mean() > scores.grad[0].abs().mean()
