import numpy as np
import torch

from src.world_model import m3w_av2_assignment_constraints as model
from src.world_model import m3w_av2_recipient_reassignment as bank


def test_conditional_loss_masks_absent_antecedents():
    logits = torch.zeros(3, 3, requires_grad=True)
    label = torch.tensor([[False, True, True], [True, False, True], [True, True, False]])
    model.loss(logits, label).backward()
    assert torch.equal(logits.grad != 0., torch.tensor([[True, False, False], [True, True, False], [True, True, True]]))


def test_no_constraint_proposal_never_has_a_release_score():
    opts = bank.assignments(2)
    ctx, actions = np.zeros((2, 3)), np.zeros((2, 2, 5))
    soft, analytic, learned = model.select(ctx, actions, np.zeros((2, 3)), opts)
    assert analytic == learned == (soft[0], None, 0)
    actions[0, 0, :2] = 2.
    actions[0, 0, 2] = .001
    actions[1, 0, :2] = 1.
    soft, analytic, learned = model.select(ctx, actions, np.array([[.9]*3, [.2]*3]), opts)
    assert soft[0] == 0
    assert analytic[0] == 1 and analytic[2] == 1
    assert learned[0] == 0 and learned[2] == 1


def test_all_constraints_replay_exact_evaluator_with_partial_labels():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 1., 3., .8])[:, None]
    valid = np.ones(cv.shape[:-1], dtype=bool)
    valid[0] = False
    valid[1, -2:] = False
    ols = cv.copy()
    p = np.stack((y*.7, y*.85), axis=1)
    opts = bank.assignments(len(cv))
    z = model.labels(dict(cv=cv, ols4=ols), p, opts, y, valid, 2.)
    exact, support = bank.eligibility(cv, ols, p, y, valid, 2., opts)
    np.testing.assert_array_equal(z[:, 0], support)
    np.testing.assert_array_equal(z.all(1), exact)
    assert exact.any() and not support.all()
