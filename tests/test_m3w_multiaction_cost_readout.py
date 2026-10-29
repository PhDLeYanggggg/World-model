import numpy as np

from src.evaluation.m3w_multiaction_cost_readout import chosen_values, matched_choices, paired_bounds, compact_score


def test_one_action_choice_and_count_matching_are_query_local():
    left = np.array([0, 1, -1, 2, -1]); right = np.array([2, -1, -1, 3, 4])
    a, b = matched_choices(left, right, np.arange(5.), np.arange(5.),
        np.array(['a']*3+['b']*2), np.array([1, 1, 1, 2, 2]), np.arange(5))
    assert a.tolist() == [-1, 1, -1, 2, -1]
    assert b.tolist() == [2, -1, -1, -1, 4]
    assert np.count_nonzero(a >= 0) == np.count_nonzero(b >= 0) == 2


def test_paired_unknown_bound_is_action_difference_not_oracle_imputation():
    floor = np.zeros((3, 12, 2)); rollouts = np.zeros((3, 2, 12, 2))
    rollouts[:, 0, :, 0] = 1; rollouts[:, 1, :, 0] = 2
    packet = dict(ade=np.array([[4., 3.], [5., 4.], [np.nan, np.nan]]),
        floor_ade=np.array([5., 6., np.nan]), floor_rollout=floor, rollouts=rollouts)
    result = paired_bounds(packet, np.array([0, 0, 0]), np.array([1, 1, 1]))
    assert result['known_delta_mass'] == 2 and result['unknown_disagreement_mass'] == 1
    assert np.isclose(result['lower_pp'], 100/11)
    same = paired_bounds(packet, np.array([0, 0, -1]), np.array([0, 0, -1]))
    assert same['lower_pp'] == same['upper_pp'] == 0


def test_empty_real_action_risk_never_passes():
    y = np.tile([1., 0., 2., 2., 0.], (2, 7, 1))
    score = compact_score(y, np.ones((2, 7)), np.array([-1, -1]))
    assert score['easy_selected_risk_upper'] is None
    assert score['empty_risk_status'] == 'not_estimable'
    assert not score['budget_status']['supported']


def test_noop_fallback_does_not_add_reference_denominators():
    candidate = np.arange(3*2*2).reshape(3, 2, 2)
    floor = np.zeros((3, 2), int)
    out = chosen_values(candidate, floor, np.array([1, -1, 0]))
    np.testing.assert_array_equal(out, np.array([[2, 3], [0, 0], [8, 9]]))
