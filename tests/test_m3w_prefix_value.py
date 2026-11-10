import numpy as np
import pytest
import torch

from scripts.run_m3w_opendd_prefix_value import saved_plan
from src.world_model.m3w_prefix_value import PrefixValue, decision_scores, objective


def test_gain_only_supervision_ignores_risk_and_illegal_actions():
    predicted = torch.zeros((2, 2, 5), requires_grad=True)
    valid = torch.tensor([[True, True], [True, False]])
    truth = torch.zeros_like(predicted)
    truth[..., 2:] = 10.
    truth[1, 1] = 20.
    loss = objective(predicted, truth, valid, risk=False)
    assert loss.item() == 0.
    loss.backward()
    assert not predicted.grad.any()
    assert objective(predicted, truth, valid, risk=True).item() > 0.


def test_penalty_is_incremental_and_fallback_is_preserved():
    prediction = np.array([[[.2, .3, -.2, .4, .1], [.5, .7, .4, -.1, .1]],
                           [[.3, .4, -.2, -.4, -.1], [.5, .7, -.4, -.1, -.1]]])
    mask = np.array([[True, True, False], [True, True, True]])
    np.testing.assert_allclose(decision_scores(prediction, np.array([2., 3.]), mask, penalty=True),
                               [[0., -.4, -np.inf], [0., .9, 1.5]])
    assert PrefixValue(7)(torch.zeros((3, 7))).shape == (3, 2, 5)


def test_matched_plan_must_obey_both_credits_and_motion():
    legal = np.tile([True, True, False], (7, 1))
    movement = np.tile([0., .1, 0.], (7, 1))
    movement[6, 1] = .2
    frozen = np.array([0, 0, 0, 0, 0, 1, 0])
    assert saved_plan(movement, legal, frozen, np.array([0, 0, 0, 0, 1, 0, 0])) == .1
    for plan in ([1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [0] * 7):
        with pytest.raises(ValueError, match='realized budget'):
            saved_plan(movement, legal, frozen, np.array(plan))
