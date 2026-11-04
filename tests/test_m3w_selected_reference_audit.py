import numpy as np
import pytest
from src.evaluation.m3w_selected_reference_audit import selected_reference


def score(y, k, c, p, scale=None):
    return selected_reference(np.array(y, float), np.array(k, bool),
        np.ones(len(y)) if scale is None else np.array(scale, float),
        np.array(c, int), dict(head=np.array(p, float)))


def test_empty_selection_keeps_unavailable_loss_not_safe_zero():
    r = score([1], [True], [-1], [2])
    assert r['observed_reference_support'] is None
    assert r['models']['head']['known_row_normalized_MSE'] is None
    assert not r['models']['head']['positive_prediction_without_observed_support']


def test_whole_unknown_is_not_an_observed_zero_future_label():
    r = score([999], [False], [6], [2])
    assert r['selected_unknown'] == 1 and r['selected_known'] == 0
    assert r['models']['head']['known_row_normalized_MSE'] is None
    assert r['models']['head']['predicted_unknown_ER_credit'] == 2


def test_known_zero_support_remains_visible_even_with_no_missing_labels():
    r = score([0, 0], [True, True], [6, 6], [1, 2])
    assert not r['observed_reference_support'] and not r['selected_unknown']
    assert r['models']['head']['positive_prediction_without_observed_support']
    assert r['models']['head']['known_record_normalized_mass_error'] == 1.5


def test_row_and_record_losses_do_not_get_confused():
    r = score([0, 2], [True, True], [6, 6], [1, 1])['models']['head']
    assert r['known_row_normalized_MSE'] == 1
    assert r['known_record_normalized_mass_squared_error'] == 0


def test_record_normalization_uses_only_selected_known_causal_scales():
    r = score([2, 999, 1], [True, False, True], [6, 6, -1], [4, 3, 10], [2, 100, 200])
    assert r['models']['head']['known_record_normalized_mass_error'] == 1
    assert r['models']['head']['predicted_full_ER_mass'] == 7


def test_invalid_labels_or_prediction_shapes_fail():
    with pytest.raises(ValueError):
        score([1], [True], [7], [2])
    with pytest.raises(ValueError):
        score([1], [True], [6], [2, 3])
