import itertools

import numpy as np
import pytest

from src.evaluation import m3w_train_feasibility as api


def brute_force(y, env, allowed):
    ids = np.flatnonzero(allowed)
    candidates = []
    for bits in itertools.product((False, True), repeat=len(ids)):
        action = np.zeros(len(y), bool)
        action[ids] = bits
        score = api.metrics.policy_score(y, action, env)
        if api.budget_status(score)['supported']:
            candidates.append(score['selected_net_gain_lower_mass'])
    return max(candidates) if candidates else None


def test_positive_oracle_and_easy_anchor_are_offline_only():
    y = np.array([[3., 0., 4., 0., 0.], [0., 0., 1., 1., 0.],
                  [0., 8., 1., 1., 8.], [np.nan]*5])
    env = np.array([3., 0., 8., 10.])
    d, action = api.solve(y, env, np.ones(4, bool))
    assert action.tolist() == [True, True, False, False]
    assert d['safe_positive_selection_exists']
    assert d['optimal_gain_lower_mass'] == 3.
    assert d['anchor_count'] == 1 and d['anchor_harm_mass'] == 0.
    assert d['label_aware_diagnostic_only'] and not d['deployable']


def test_harmful_easy_anchor_can_be_required_and_is_minimum_cost():
    y = np.array([[1., 0., 2., 0., 0.], [0., .01, 1., 1., .01],
                  [0., .015, 2., 2., .015], [0., .005, .1, .1, .005]])
    d, action = api.solve(y, np.ones(4), np.ones(4, bool))
    assert action.tolist() == [True, True, False, False]
    assert d['optimal_gain_lower_mass'] == pytest.approx(.99)
    assert d['anchor_harm_mass'] == .01


@pytest.mark.parametrize('y,reason', [
    ([[1., 0., 2., 0., 0.]], 'no_allowed_known_easy_reference'),
    ([[1., 0., 2., 0., 0.], [0., .3, 1., 1., .3]], 'no_easy_anchor_within_budget'),
    ([[np.nan]*5], 'no_allowed_known_easy_reference'),
])
def test_positive_headroom_does_not_imply_supported_safety(y, reason):
    y = np.array(y)
    d, action = api.solve(y, np.ones(len(y)), np.ones(len(y), bool))
    assert not d['budget_feasible'] and d['optimal_gain_lower_mass'] is None
    assert d['infeasibility_reason'] == reason and not action.any()
    assert d['policy']['easy_selected_risk_upper'] is None


def test_safe_negative_gain_is_not_positive_headroom():
    y = np.array([[0., .01, 1., 1., .01]])
    d, _ = api.solve(y, np.ones(1), np.ones(1, bool))
    assert d['budget_feasible'] and not d['safe_positive_selection_exists']
    assert d['optimal_gain_lower_mass'] == -.01


def test_eligibility_and_deletion_only_cannot_be_bypassed():
    y = np.array([[2., 0., 2., 2., 0.], [0., .5, 1., 1., .5]])
    d, action = api.solve(y, np.array([2., 1.]), np.array([False, True]))
    assert not d['budget_feasible'] and not action.any()
    assert d['allowed_rows'] == 1 and d['known_positive_rows'] == 0
    with pytest.raises(ValueError): api.solve(y, np.ones(2), np.ones(2))


def test_closed_form_matches_exhaustive_boolean_search():
    rng = np.random.default_rng(913)
    for _ in range(240):
        n = 6
        ref = rng.choice([0., .2, 1., 4.], size=n)
        easy = rng.random(n) < .5
        delta = rng.choice([0., .01, .03, .4, -1., -.01, -.03, -.8], size=n)
        b = np.minimum(np.maximum(delta, 0.), ref)
        h = np.maximum(-delta, 0.)
        y = np.column_stack((b, h, ref, ref*easy, h*easy))
        env = b+h+.1
        y[rng.random(n) < .15] = np.nan
        allowed = rng.random(n) < .8
        d, action = api.solve(y, env, allowed)
        expected = brute_force(y, env, allowed)
        assert d['budget_feasible'] == (expected is not None)
        if expected is not None:
            assert d['optimal_gain_lower_mass'] == pytest.approx(expected, abs=1e-12)
            assert api.budget_status(api.metrics.policy_score(y, action, env))['supported']
        assert not (action & ~allowed).any()


def test_unknown_envelope_never_becomes_a_zero_label():
    y = np.array([[.2, 0., 1., 1., 0.], [np.nan]*5])
    d, action = api.solve(y, np.array([1., 100.]), np.ones(2, bool))
    assert action.tolist() == [True, False]
    assert d['unknown_allowed_rows'] == 1
    assert d['policy']['unknown_rows'] == 1
    assert d['policy']['selected_unknown'] == 0


@pytest.mark.parametrize('recordings', [np.array(['a', 'a', 'b']), np.array([1, 1, 2])])
def test_record_view_keeps_unidentifiable_record_and_frozen_actions(recordings):
    y = np.array([[.2, 0., 1., 1., 0.], [0., .7, 1., 1., .7],
                  [.3, 0., 1., 0., 0.]])
    args = (np.zeros((3, 380)), np.ones(3), y, None, None,
            recordings, np.arange(3), {})
    action = np.array([True, True, True])
    copy = action.copy()
    d = api.evaluate(args, np.ones(3, bool), {'frozen': action})
    assert np.array_equal(copy, action)
    first, second = sorted(set(map(str, recordings)))
    assert set(d['recordings']) == {first, second}
    assert not d['all_recordings_feasible']
    assert d['recordings'][first]['oracle']['optimal_gain_lower_mass'] == .2
    assert not d['recordings'][second]['oracle']['budget_feasible']
    assert d['recordings'][first]['policies']['frozen']['observed']['known_rows'] == 2
    assert d['recordings'][second]['policies']['frozen']['observed']['known_rows'] == 1
    assert d['policies']['frozen']['deletion_oracle']['budget_feasible']
    assert not d['policies']['frozen']['all_record_deletion_feasible']
