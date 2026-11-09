import numpy as np
import torch

from src.world_model import m3w_av2_assignment_order as model
from src.world_model import m3w_av2_recipient_reassignment as bank


def test_continuous_target_respects_exact_qualification_with_missing_rows():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 1., 3., .8])[:, None]
    valid = np.ones(cv.shape[:-1], dtype=bool)
    valid[0] = False
    valid[1, -2:] = False
    p = np.stack((y*.7, y*.85), axis=1)
    opts = bank.assignments(len(cv))
    score, ok = model.targets(dict(cv=cv, ols4=cv), p, opts, y, valid, 2., 1.)
    expected, _ = bank.eligibility(cv, cv, p, y, valid, 2., opts)
    np.testing.assert_array_equal(ok, expected)
    assert score[ok].min() > score[~ok].max()
    assert len(np.unique(score[ok])) > 1


def test_padded_positions_have_no_gradient_and_scene_permutation_is_invariant():
    pred = torch.tensor([[0., .2, 1000.], [.4, .1, .5]], requires_grad=True)
    y = torch.tensor([[1., -.3, -1000.], [0., .2, .1]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    a = model.losses(pred, y, valid)
    b = model.losses(pred.flip(1), y.flip(1), valid.flip(1))
    torch.testing.assert_close(torch.stack(a), torch.stack(b))
    sum(a).backward()
    assert pred.grad[0, 2] == 0.


def test_listwise_loss_prefers_order_within_a_scenario():
    target = torch.tensor([[1.2, .1, -.4]])
    mask = torch.ones_like(target, dtype=torch.bool)
    assert model.losses(target, target, mask)[1] < model.losses(target.flip(1), target, mask)[1]
    assert model.choose(np.array([-.4, -.1, -.3]))[0] == 1
