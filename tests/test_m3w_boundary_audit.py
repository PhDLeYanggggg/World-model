import numpy as np
import pytest
from src.evaluation.m3w_boundary_audit import summarize


def test_mse_improvement_without_decision_change_and_undefined_easy():
    y = np.array([[-1., 0.], [-2., 0.]])
    old = np.array([[3., 1.], [4., 2.]])
    new = np.array([[2., .1], [3., .2]])
    m = np.array([[0., 0., 50., 0., 0.], [0., 0., 100., 0., 0.]])
    a = summarize(y, old, new, m, np.ones(2), np.ones(2))
    assert a['strata']['both_reject']['rows'] == 2
    assert a['weighted_MSE_reduction'] > 0
    assert a['decisions']['new']['accept']['selected_easy_risk'] is None
    assert a['decisions']['new']['both']['true_joint_safe_rows'] == 2
    assert not a['population_safety_certificate']


def test_rejection_axes_and_mse_decomposition():
    y = np.array([[-1., -.5], [1., 2.], [-1., 0.], [-1., -.5]])
    old = np.array([[1., -1.], [-1., 1.], [1., 1.], [-1., -1.]])
    new = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., -1.]])
    m = np.ones((4, 5))
    a = summarize(y, old, new, m, np.ones(4), np.ones(2))
    assert a['decisions']['old']['all_only']['rows'] == 1
    assert a['decisions']['old']['easy_only']['rows'] == 1
    assert a['decisions']['new']['accept']['true_joint_unsafe_rows'] == 1
    assert sum(v['weighted_MSE_reduction_contribution'] for v in a['strata'].values()) == pytest.approx(a['weighted_MSE_reduction'])


def test_reject_invalid_scale():
    with pytest.raises(ValueError):
        summarize(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)),
            np.zeros((1, 5)), np.ones(1), np.zeros(2))
