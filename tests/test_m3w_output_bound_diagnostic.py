import numpy as np
import pytest
from src.evaluation.m3w_output_bound_diagnostic import bound_relaxation


def test_closed_ball_infimum_and_finite_cv_are_separate():
    baseline = np.zeros((3, 2, 2))
    budget = np.array([[1., 1.], [0., 0.], [1., 1.]])
    cv = np.zeros_like(baseline)
    cv[2, :, 0] = 1.
    target = np.zeros_like(baseline)
    target[:, :, 0] = 3.
    valid = np.ones((3, 2), bool)
    result = bound_relaxation(baseline, budget, cv, target, valid)
    np.testing.assert_equal(result['independent_step_ball_ADE_infimum'], [2., 3., 2.])
    np.testing.assert_equal(result['finite_CV_geometrically_representable'], [True, True, False])
    np.testing.assert_equal(result['CV_boundary_or_numerically_ambiguous'], [False, False, True])
    assert not result['CV_outside_bound'].any()


def test_missing_future_is_not_zero_or_a_causal_geometry_input():
    base = np.zeros((2, 2, 2)); cv = base.copy(); cv[1, :, 0] = 2.
    target = np.full_like(base, np.nan); valid = np.zeros((2, 2), bool)
    result = bound_relaxation(base, np.ones((2, 2)), cv, target, valid)
    assert np.isnan(result['independent_step_ball_ADE_infimum']).all()
    np.testing.assert_equal(result['CV_outside_bound'], [False, True])
    with pytest.raises(ValueError):
        bound_relaxation(base, -np.ones((2, 2)), cv, target, valid)
