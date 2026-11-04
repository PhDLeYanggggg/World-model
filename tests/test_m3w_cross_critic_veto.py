import numpy as np
import pytest
from src.world_model.m3w_cross_critic_veto import choices, rollout


def packet():
    p = np.tile([5., .01, 10., 1., .001], (4, 7, 1)).astype(float)
    p[:, 6, 0] = [8., 9., 7., 6.]
    q = p.copy()
    allowed = np.ones((4, 7), bool)
    allowed[:, 0] = False
    return p, q, allowed, [np.array([0, 1]), np.array([2, 3])], np.array([9, 8, 7, 6])


def test_veto_does_not_reselect_and_control_matches_every_query():
    p, q, allowed, groups, ids = packet()
    q[1, 6, 1] = 1.
    q[3, 6, 4] = .1
    original = p.copy(), q.copy(), allowed.copy()
    result = choices(p, q, allowed, groups, ids)
    np.testing.assert_array_equal(result['cross_veto'], [6, -1, 6, -1])
    np.testing.assert_array_equal(result['self_count_matched'], [-1, 6, 6, -1])
    assert result['frozen_self'][1] != -1
    for group in groups:
        assert (result['cross_veto'][group] >= 0).sum() == (result['self_count_matched'][group] >= 0).sum()
    for a, b in zip(original, (p, q, allowed)):
        np.testing.assert_array_equal(a, b)


def test_zero_reference_and_empty_queries_are_not_safe_events():
    p, q, allowed, groups, ids = packet()
    q[:, :, 3] = 0
    result = choices(p, q, allowed, groups, ids)
    assert np.all(result['cross_veto'] == -1)
    assert np.all(result['self_count_matched'] == -1)
    assert 'safety_pass' not in result


def test_causal_ties_use_row_identity_not_array_order():
    p, q, allowed, groups, ids = packet()
    p[0, 6, 0] = p[1, 6, 0]
    q[1, 6, 1] = 1.
    result = choices(p, q, allowed, groups, ids)
    assert result['self_count_matched'][1] == 6
    with pytest.raises(TypeError):
        choices(p, q, allowed, groups, ids, future_endpoint=np.zeros((4, 2)))


def test_incomplete_queries_nonfinite_values_and_duplicate_ids_fail():
    p, q, allowed, groups, ids = packet()
    for g, row_ids in (([np.array([0, 1]), np.array([1, 2, 3])], ids), (groups, np.zeros(4, int))):
        with pytest.raises(ValueError):
            choices(p, q, allowed, g, row_ids)
    q[0, 6, 0] = np.nan
    with pytest.raises(ValueError):
        choices(p, q, allowed, groups, ids)


def test_fallback_is_exact_floor_and_empty_control_is_unavailable():
    from src.evaluation.m3w_matched_intervention_utility import matched_utility
    floor = np.ones((2, 12, 2))
    paths = np.ones((2, 7, 12, 2))*3
    selected = np.array([-1, -1])
    actual = rollout(paths, floor, selected)
    np.testing.assert_array_equal(actual, floor)
    score = matched_utility(actual, actual, floor, floor, np.ones((2, 12), bool),
        selected, selected, [np.array([0, 1])])
    assert score['contrast']['known_pp'] is None and not score['safety_gate_usage']
