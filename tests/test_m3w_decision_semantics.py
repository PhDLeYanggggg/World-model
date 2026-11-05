import numpy as np
import pytest
from src.evaluation.m3w_decision_semantics import proposal_mask, joint_target, diagnose


def test_exact_one_action_support_and_fallback():
    legal = np.array([[True, True], [False, True], [True, True]])
    np.testing.assert_array_equal(proposal_mask(np.array([1, 1, -1]), legal),
        np.array([[False, True], [False, True], [False, False]]))
    with pytest.raises(ValueError):
        proposal_mask(np.array([0, 0, -1]), legal)
    with pytest.raises(ValueError):
        proposal_mask(np.array([0., 1., -1.]), legal)


def test_joint_sign_handles_opposing_channels_and_zero():
    y = np.array([[-100., .1], [-.2, 0.], [0., 0.], [.5, .2]])
    np.testing.assert_allclose(joint_target(y), [.1, -.2, 0., .7])
    np.testing.assert_array_equal(joint_target(y) <= 0, (y <= 0).all(1))


def test_diagnosis_masks_unknown_and_separates_joint_costs():
    y = np.array([[[-100., .1]], [[-.2, 0.]], [[1000., 1000.]]])
    r = np.array([[[-1., -1.]], [[1., -1.]], [[-1., -1.]]])
    d = r.copy()
    result = diagnose(r, d, y, np.ones((3, 1), bool), np.array([True, True, False]), np.ones(3))
    assert result['false_safe_count'] == result['false_reject_count'] == 1
    assert result['total_joint_cost'] == pytest.approx(.3)
    assert result['mixed_sign_proposals'] == 1
    assert result['nonjoint_channel_reward_mass_on_joint_unsafe'] == 100.
    assert result['unknown_proposals'] == 1
    assert not result['population_risk_certificate']
