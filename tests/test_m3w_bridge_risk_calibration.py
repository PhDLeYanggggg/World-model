import numpy as np
import pytest
from src.evaluation.m3w_bridge_risk_calibration import apply, evidence, fit, zero_violation_scene_upper, hb_zero_loss_floor


def test_easy_event_is_CV_defined_but_denominator_is_delivered_reference():
    cv = np.array([1., 50.]); r = np.array([40., 1.]); p = r+np.array([2., 0.])
    e = evidence(np.ones(2, bool), cv, r, p, np.array(['a', 'a']), easy_cut=2.)
    event = e['by_locality']['a']['events']['easy']
    assert event['rows'] == 1 and event['reference_mass'] == 40 and event['ratio'] == .05
    assert not e['feasible']


def test_unknown_future_never_changes_inference_or_disappears_from_action_count():
    u = np.tile([2., 1.], (2, 1)); m = np.array([[100., 1.], [100., 1.]])
    bits = apply(u, m, np.ones(2, bool), np.ones(2), dict(kind='none', abstain=False))
    r = np.array([1., np.nan]); p = np.array([.5, np.nan])
    e = evidence(bits, r, r, p, np.array(['s', 's']), easy_cut=2.)
    assert e['unknown_selected'] == 1 and e['selected'] == 2
    with pytest.raises(TypeError): apply(u, m, np.ones(2, bool), np.ones(2), dict(kind='none', abstain=False), future=r)


def test_tightening_preserves_causal_guards_and_rejects_relaxation():
    u = np.tile([2., 1.], (3, 1)); m = np.array([[100., 1.], [100., 1.], [100., 1.]])
    move, env = np.array([0, 1, 1], bool), np.array([1., 0., 1.])
    rule = dict(kind='population_rescale', abstain=False, harm_multiplier=4., denominator_multiplier=1.)
    assert not apply(u, m, move, env, rule).any()
    with pytest.raises(ValueError): apply(u, m, move, env, dict(kind='selected_risk_grid', abstain=False, threshold=.03))


def test_grid_selects_safe_gain_and_preserves_negative_results():
    u = np.tile([2., 1.], (4, 1)); m = np.array([[100., .1], [100., 1.]]*2)
    r = np.ones(4); p = np.array([.5, 2., .5, 2.]); s = np.array(['a', 'a', 'b', 'b'])
    fitted = fit(u, m, np.ones(4, bool), np.ones(4), r, r, p, s, easy_cut=2., grid=[0., .001, .02])
    assert fitted['rules']['selected_risk_grid']['threshold'] == .001
    assert not fitted['evidence']['none']['feasible']
    assert fitted['evidence']['selected_risk_grid']['feasible']
    assert not fitted['calibrated_guarantee']


def test_empty_easy_support_is_undefined_not_pass():
    r = np.array([100.]); e = evidence(np.array([False]), r, r, r, np.array(['s']), easy_cut=1.)
    assert e['by_locality']['s']['events']['easy']['ratio'] is None and not e['feasible']


def test_zero_threshold_is_not_structural_abstention():
    u = np.array([[2., 1.]]); m = np.array([[1., 0.]])
    assert apply(u, m, np.array([True]), np.ones(1), dict(kind='selected_risk_grid', threshold=0., abstain=False))[0]
    assert not apply(u, m, np.array([True]), np.ones(1), dict(kind='selected_risk_grid', threshold=None, abstain=True))[0]


def test_binomial_and_HB_illustrations_do_not_count_overlapping_rows_as_scenes():
    assert np.isclose(zero_violation_scene_upper(12, .05), 1-.05**(1/12))
    assert zero_violation_scene_upper(12, .05) > .22
    assert hb_zero_loss_floor(12, .02) > .78
    assert zero_violation_scene_upper(6, .05) > zero_violation_scene_upper(12, .05)
    with pytest.raises(ValueError): zero_violation_scene_upper(0, .05)
