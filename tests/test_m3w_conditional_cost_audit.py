import numpy as np
import pytest
from src.evaluation.m3w_conditional_cost_audit import ARMS, strict_bits, fixed_groups, conditional_stats


def fixture():
    scores = {a:np.array([[2., .1], [2., .1], [1., .2], [1., .05]]) for a in ARMS}
    y = np.array([[0., 2.], [4., 0.], [np.nan, np.nan], [np.nan, np.nan]])
    return scores, y, np.array([True, True, False, False]), np.array([True, True, True, False])


def test_same_support_and_harm_frequency_severity_identity():
    scores, y, full, any_label = fixture()
    result = conditional_stats(scores, y, full, any_label, np.ones(4, bool), np.ones(4)*4)
    assert (result['rows'], result['complete'], result['unknown_ADE'], result['incomplete']) == (4, 2, 1, 2)
    c = result['costs']
    assert c['harm_frequency']*c['harm_given_event'] == c['realized_harm'] == 1
    assert c['realized_net_gain'] == 1 and c['realized_fractional_harm'] == .25
    assert c['native'] == c['fraction'] == c['tempered']


def test_future_outcomes_cannot_change_past_only_membership():
    scores, y, full, any_label = fixture()
    past = np.ones((4, 8, 2)); past[:, -1] = 2
    d = np.ones(4)*4
    bits = {a:strict_bits(p, past, d) for a,p in scores.items()}
    masks = fixed_groups(bits, np.ones(4, bool))
    first = conditional_stats(scores, y, full, any_label, masks['tempered_selected'], d)
    y[:2] = [[9, 0], [0, 10]]
    later = conditional_stats(scores, y, full, any_label, masks['tempered_selected'], d)
    assert first['rows'] == later['rows'] == 3
    assert first['costs'] != later['costs']
    for a in ARMS:
        np.testing.assert_array_equal(bits[a], strict_bits(scores[a], past, d))


def test_empty_cost_support_is_unknown_not_safe():
    scores, y, full, any_label = fixture()
    result = conditional_stats(scores, y, full, any_label, ~full, np.ones(4))
    assert result['costs'] is None and result['complete'] == 0


def test_groups_are_frozen_and_disjoint_pair_differences():
    bits = dict(native=np.array([True, False, False]), fraction=np.array([False, True, False]),
                tempered=np.array([True, False, True]))
    groups = fixed_groups(bits, np.ones(3, bool))
    np.testing.assert_array_equal(groups['tempered_not_native'], [False, False, True])
    np.testing.assert_array_equal(groups['native_not_tempered'], [False, False, False])
    np.testing.assert_array_equal(groups['union'], [True, True, True])


def test_invalid_or_outcome_unsupported_inputs_rejected():
    scores, y, full, any_label = fixture()
    with pytest.raises(ValueError):
        conditional_stats(scores, y, full, np.zeros(4, bool), np.ones(4, bool), np.ones(4))
    with pytest.raises(ValueError):
        fixed_groups({a:np.ones(4, bool) for a in ARMS}, np.zeros(4, bool))
    with pytest.raises(ValueError):
        strict_bits(scores['native'], np.ones((4, 7, 2)), np.ones(4))
