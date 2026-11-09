import numpy as np
import pytest

from scripts.audit_m3w_ols_relative_gain_ceiling import exact_count_upper
from scripts.audit_m3w_ols_predicted_feasibility import relaxed_excess
from src.world_model.m3w_ols_relative_gain import budgeted, decisions, paired_targets


def test_anchor_changes_signed_gain_not_cv_risk_or_unknown_mask():
    p = np.zeros((2, 5, 12, 2))
    p[..., 0] = np.array([4, 2, 1, 3, 5])[None, :, None]
    y = np.zeros((2, 12, 2))
    v = np.array([[True]*12, [False]*12])
    cv, ols, ref = paired_targets(p, y, v, np.ones(2), 5.)
    np.testing.assert_array_equal(cv[..., 1:], ols[..., 1:])
    np.testing.assert_array_equal(ref, [[4., 4.], [0., 0.]])
    np.testing.assert_array_equal(cv[..., 0], [[2., 3., 1., -1.], [-2., -3., -1., -1.]])
    np.testing.assert_array_equal(ols[..., 0], [[0., 1., -1., -3.], [0., -1., -1., -3.]])


def test_joint_budget_includes_cost_of_unreplaced_ols():
    gain = np.array([[0., .5], [0., 0.]])
    legal = np.array([[False, True], [False, False]])
    changed = np.ones_like(legal)
    refs = np.full((2, 2), 10.)
    harm = np.array([[[.4, .4], [0., 0.]], [[1., 1.], [0., 0.]]])
    x, ok = budgeted(gain, harm, refs, changed, legal)
    assert not ok
    np.testing.assert_array_equal(x, [0, 0])
    harm[1, 0] = .2
    x, ok = budgeted(gain, harm, refs, changed, legal)
    assert ok
    np.testing.assert_array_equal(x, [1, 0])


def test_matched_counts_are_replacements_of_ols_not_cv():
    p = np.zeros((2, 5, 12, 2))
    p[..., 0] = np.array([0, 1, 2, 3, 4])[None, :, None]
    c, o = np.zeros((2, 4, 3)), np.zeros((2, 4, 3))
    c[0, 1, 0] = 1.
    o[1, 2, 0] = 3.
    x, ok = decisions(p, c, o, np.ones((2, 2)))
    assert ok
    np.testing.assert_array_equal(x[:, 0], [1, 0])
    np.testing.assert_array_equal(x[:, 1], [0, 2])
    assert np.count_nonzero(x[:, 3]) == np.count_nonzero(x[:, 4])


def test_exact_count_ceiling_keeps_forced_losses_and_excludes_unchanged_paths():
    options = np.array([[3., 2.], [-np.inf, -.5], [-np.inf, -np.inf]])
    np.testing.assert_array_equal(exact_count_upper(options, np.array([[0, 2], [1, 2]])),
                                  [[0., 2.5], [3., 2.5]])
    with pytest.raises(ValueError, match="Infeasible"):
        exact_count_upper(options, np.array([3]))


def test_relaxed_risk_bound_optimizes_each_constraint_separately():
    changed = np.array([[True, True], [False, True]])
    harm = np.array([[[.6, .1], [.1, .6]], [[99., 99.], [.3, .3]]])
    baseline, lower = relaxed_excess(changed, harm, np.full((2, 2), 10.))
    np.testing.assert_allclose(baseline, [.4, -.1])
    np.testing.assert_allclose(lower, [-.1, -.1])
