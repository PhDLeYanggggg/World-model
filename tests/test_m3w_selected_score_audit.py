import numpy as np
import pytest
from src.evaluation.m3w_selected_score_audit import audit


def fixture():
    p = np.broadcast_to([.2, .01, .5, .2, .001], (4, 7, 5)).copy()
    y = np.broadcast_to([.1, .04, .5, .2, .02], (4, 7, 5)).copy(); y[2] = np.nan
    return p, y, np.array([1., 2., 3., 4.]), np.ones((4, 7), bool), [np.array([0, 1]), np.array([2, 3])]


def test_query_error_splits_into_harm_and_reference_bias_with_correct_sign():
    p, y, scale, allowed, groups = fixture(); p[:, :, 2] = .8
    r = audit(p, y, scale, allowed, groups, np.array([0, 0, -1, -1]), np.array([0, 0, -1, -1]))
    np.testing.assert_allclose(r['mean_harm_underprediction'], [.02, .038/3])
    np.testing.assert_allclose(r['mean_reference_bias_contribution'], [.004, 0.])
    np.testing.assert_allclose(r['mean_signed_score_underprediction'], [.024, .038/3])
    assert r['complete_selected_queries'] == 1 and r['empty_selected_queries'] == 1
    assert not r['inference_feature'] and not r['risk_certificate']


def test_unknown_and_empty_queries_are_retained_not_zero_supervision():
    p, y, scale, allowed, groups = fixture()
    initial = np.array([0, 0, -1, -1]); choice = np.array([1, 0, 1, -1])
    r = audit(p, y, scale, allowed, groups, initial, choice)
    assert r['queries'] == 2 and r['rows'] == 4
    assert r['complete_selected_queries'] == 1 and r['unknown_selected_queries'] == 1
    assert r['selected_unknown_rows'] == 1 and r['introduced_rows'] == 1
    assert r['changed_action_rows'] == 1 and r['changed_rows'] == 2
    assert r['predicted_signed_safe_observed_positive_excess'] == [1, 1]


def test_all_empty_support_is_not_estimable():
    p, y, scale, allowed, groups = fixture(); empty = np.full(4, -1)
    r = audit(p, y, scale, allowed, groups, empty, empty)
    assert r['empty_selected_queries'] == 2 and r['complete_selected_queries'] == 0
    assert r['mean_signed_score_underprediction'] is None and r['mean_score_squared_error'] is None


def test_zero_reference_is_undefined_support_not_risk_certification():
    p, y, scale, allowed, groups = fixture(); y[:2, :, 2:4] = 0
    r = audit(p, y, scale, allowed, groups, np.zeros(4, int), np.array([0, 0, -1, -1]))
    assert r['zero_true_reference_complete_queries'] == [1, 1]
    assert not r['risk_certificate']


def test_changed_shapes_partial_labels_or_illegal_actions_refuse():
    p, y, scale, allowed, groups = fixture(); choice = np.array([0, 0, -1, -1])
    bad = y.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError): audit(p, bad, scale, allowed, groups, choice, choice)
    allowed[0, 0] = False
    with pytest.raises(ValueError): audit(p, y, scale, allowed, groups, choice, choice)
