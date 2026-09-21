import numpy as np
import pytest

from src.evaluation.m3w_cost_fit_forensics import conditional_cost_summary, diagnose_fit

RULES = {'fixed': {'min_predicted_gain': .02, 'max_agent_predicted_harm': .05}}


def test_zero_clipped_prediction_is_not_zero_realized_harm():
    y = np.array([[0., .4], [0., 10.]])
    raw = np.array([[.5, -.2], [0., 10.]])
    r = diagnose_fit(y, np.maximum(raw, 0), ['a', 'b'], RULES, raw_ridge=raw)
    clipped = r['conditional']['ridge_harm_clipped_to_zero']
    assert clipped['predicted_harm_mean'] == 0 and clipped['target_harm_mean'] == .4
    eligible = r['conditional']['eligible_fixed']
    assert eligible['rows'] == 1 and eligible['actual_net_gain_mean'] == -.4
    assert r['full_fit']['harm_mse'] < r['mean_label_reference']['harm_mse']
    assert r['per_recording']['b']['eligible_fixed']['rows'] == 0
    assert r['per_agent_eligibility_is_not_scene_intervention']


def test_empty_eligibility_retains_unknown_not_zero():
    r = diagnose_fit(np.array([[0., 0.]]), np.array([[0., .1]]), ['r'], RULES)
    assert r['conditional']['eligible_fixed']['target_harm_mean'] is None
    assert r['top_harm_squared_target_mass_fraction'] is None


def test_no_oracle_label_used_for_membership():
    pred = np.array([[.1, .01], [.2, .2]])
    a = diagnose_fit(np.array([[0., 1.], [1., 0.]]), pred, ['a','b'], RULES)
    b = diagnose_fit(np.array([[10., 0.], [0., 10.]]), pred, ['a','b'], RULES)
    assert a['conditional']['eligible_fixed']['rows'] == b['conditional']['eligible_fixed']['rows'] == 1
    assert a['conditional']['eligible_fixed']['predicted_net_gain_mean'] == b['conditional']['eligible_fixed']['predicted_net_gain_mean']


def test_reject_ambiguous_target_order_or_changed_clipping():
    with pytest.raises(ValueError, match='both'):
        diagnose_fit(np.ones((1,2)), np.ones((1,2)), ['r'], RULES)
    with pytest.raises(ValueError, match='reconstruct'):
        diagnose_fit(np.zeros((1,2)), np.ones((1,2)), ['r'], RULES, raw_ridge=np.zeros((1,2)))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_reject_invalid_costs(bad):
    with pytest.raises(ValueError):
        conditional_cost_summary([[0., bad]], [[0., 1.]], np.array([True]))


def test_boolean_membership_and_empty_population_requirements():
    with pytest.raises(ValueError):
        conditional_cost_summary([[0., 1.]], [[0., 1.]], [1])
    with pytest.raises(ValueError):
        diagnose_fit(np.zeros((0,2)), np.zeros((0,2)), [], RULES)
