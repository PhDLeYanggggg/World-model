import numpy as np
import pytest
from src.world_model.m3w_score_support_calibration import (
    role_pairs, support_distance, support_limit, static_guard, decide, metrics, calibrate)


def test_roles_exclude_entire_fitting_chain():
    pairs = role_pairs(list('abcd'), list('efgh'), list('ijkl'))
    assert len(pairs) == 6
    for cal, held in pairs:
        assert not set(cal)&set(held) and len(cal) == len(held) == 2
        assert not (set(cal)|set(held)) & set('abcdefgh')
    with pytest.raises(ValueError): role_pairs(list('abcd'), list('efgh'), list('hijk'))


def test_support_uses_training_weights_not_unseen_extremes():
    d = support_distance([[0, 0], [2, 2], [10000, 10000]], [0, 0], [1, 1])
    assert support_limit(d, [1, 1, 0]) == 2
    assert d[-1] > support_limit(d, [1, 1, 0])
    with pytest.raises(ValueError): support_distance([[1]], [0], [0])


def test_static_stationary_and_easy_gain_guards():
    u = np.array([[2., 1], [2, 1], [1, 2], [2, 1]])
    e = np.array([[1., 0], [1, 0], [1, 0], [1, .03]])
    g = static_guard(u, e, np.array([False, True, True, True]))
    np.testing.assert_array_equal(g, [False, True, False, False])
    np.testing.assert_array_equal(decide(np.full(4, -.1), g, np.ones(4, bool), 0), g)
    assert not decide(np.full(4, -.1), g, np.ones(4, bool), None).any()


def example():
    return dict(score=np.tile([-.1, -.01], 2), guard=np.ones(4, bool), supported=np.ones(4, bool),
        cv=np.full(4, 10.), error=np.tile([8., 11.], 2), sites=np.repeat(['cal1', 'cal2'], 2),
        calibration_sites=['cal1', 'cal2'], thresholds=[-.1, -.05, 0], easy_cut=10., hard_cut=10., min_selected=1)


def test_calibration_rejects_harm_and_threshold_is_applied_without_labels():
    r = calibrate(**example())
    assert r['threshold'] == -.1 and not r['certificate']
    decision = decide(np.array([-.2, -.01]), np.ones(2, bool), np.ones(2, bool), r['threshold'])
    np.testing.assert_array_equal(decision, [True, False])


def test_held_labels_cannot_enter_calibration():
    kw = example(); kw['sites'][0] = 'held'
    with pytest.raises(ValueError): calibrate(**kw)


def test_complete_fallback_does_not_invent_risk_or_gain():
    kw = example(); kw['error'] = kw['cv']+1
    assert calibrate(**kw)['threshold'] is None
    r = metrics([1, 0], [2, 1], np.zeros(2, bool), 1, 1)
    assert r['all_gain_percent'] == 0 and r['selected_positive_harm_ratio'] is None


def test_zero_reference_harm_not_removed_by_easy_definition():
    r = metrics([0, 10, np.nan], [1, 5, np.nan], np.ones(3, bool), 10, 10)
    assert r['zero_reference_harmed'] == 1 and r['zero_reference_harm'] == 1
    assert r['unknown'] == 1 and r['easy_gain_percent'] == 50
    assert r['selected_positive_harm_ratio'] == .1


def test_safety_required_at_each_calibration_source_not_pooled():
    kw = example(); kw['error'] = [8, 8, 12, 12]
    assert calibrate(**kw)['threshold'] is None


def test_unknown_future_does_not_change_causal_action():
    score = np.array([-.1, -.1]); take = decide(score, np.ones(2, bool), np.ones(2, bool), 0)
    a = metrics([1, np.nan], [.5, np.nan], take, 1, 1)
    b = metrics([1, 100], [.5, 200], take, 1, 1)
    assert a['unknown'] == 1 and b['unknown'] == 0
    np.testing.assert_array_equal(take, [True, True])


def test_invalid_future_masks_and_thresholds_fail_closed():
    with pytest.raises(ValueError): metrics([1, np.nan], [1, 2], np.ones(2, bool), 1, 1)
    with pytest.raises(ValueError): decide([np.nan], np.ones(1, bool), np.ones(1, bool), 0)
    with pytest.raises(ValueError): decide([0.], np.ones(1, bool), np.ones(1, bool), .1)
