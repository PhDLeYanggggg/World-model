import numpy as np

from src.evaluation.m3w_recording_blend_feasibility import audit, min_affine_cost, safe_segment


def test_min_affine_cost_masks_labels_and_handles_norm_ties():
    r = np.array([[[-1., 0.], [1000., 0.]], [[-1., 0.], [3., 0.]]])
    d = np.array([[[2., 0.], [500., 0.]], [[2., 0.], [-2., 0.]]])
    valid = np.array([[True, False], [False, False]])
    alpha, lower, upper = min_affine_cost(r, d, valid)
    np.testing.assert_allclose(alpha, [.5, 1.], atol=1e-10)
    np.testing.assert_allclose(lower, [0., 1.], atol=1e-10)
    np.testing.assert_allclose(upper, [0., 1.], atol=1e-10)


def test_feasible_witness_and_segment_are_checked_at_original_budget():
    cv = np.zeros((2, 2, 2))
    y = np.ones_like(cv)
    valid = np.array([[True, True], [False, False]])
    switch = np.ones(2, dtype=bool)
    a = np.ones_like(cv)*3.
    b = np.zeros_like(cv)-1.
    info, p = audit(cv, a, b, y, valid, switch, 'r', 2., 4.)
    assert info['status'] == 'checked_feasible'
    assert info['risk']['supported']
    assert info['identity_nudge'] > 0.
    assert (p != cv).any((1, 2)).all()
    t, path = safe_segment(p, a, cv, y, valid, switch, 'r', 2., 4.)
    assert 0. < t < 1.
    assert (path != cv).any((1, 2)).all()


def test_missing_support_and_unobserved_debit_are_distinct():
    cv = np.zeros((2, 2, 2))
    y = np.ones_like(cv)
    valid = np.array([[True, True], [False, False]])
    switch = np.ones(2, dtype=bool)
    a = np.ones_like(cv)
    b = np.ones_like(cv)*2.
    missing, p = audit(cv, a, b, y, valid, switch, 'r', 1., 4.)
    assert p is None and missing['status'] == 'structural_support_absent'
    infeasible, p = audit(cv, a, b, y, valid, switch, 'r', 2., 4.)
    assert p is None and infeasible['status'] == 'risk_infeasible_with_fixed_recipients'
    assert infeasible['unknown_radius_alone_excludes_easy']


def test_known_harm_can_exclude_even_without_missing_labels():
    cv = np.zeros((1, 2, 2))
    y = np.ones_like(cv)
    valid = np.ones((1, 2), dtype=bool)
    info, p = audit(cv, 3*y, 4*y, y, valid, np.ones(1, dtype=bool), 'r', 2., 4.)
    assert p is None and info['status'] == 'risk_infeasible_with_fixed_recipients'
    assert info['excluding_channels'] == [True, True, True]
    assert not info['unknown_radius_alone_excludes_easy']
