import numpy as np
import pytest

from src.evaluation.m3w_source_risk_calibration import apply_calibration, fit_calibration, calibration_evidence


def arrays():
    return (np.ones(8), np.tile([1., .01], (8, 1)), np.ones(8, bool), np.ones(8),
        np.full(8, .8), np.repeat(['a', 'b'], 4))


def test_calibration_accepts_benefit_without_harm():
    u, m, moving, r, p, s = arrays()
    got = fit_calibration(u, m, moving, r, p, s, easy_cut=1, event='easy', grid=[0., .01, .02])
    assert got['rules']['selected_risk_grid']['threshold'] == .01
    assert apply_calibration(u, m, moving, got['rules']['selected_risk_grid']).all()
    assert not got['calibrated_guarantee']


def test_one_locality_harm_cannot_hide_in_positive_average():
    u, m, moving, r, p, s = arrays()
    p[:4], p[4:] = .2, 1.1
    got = fit_calibration(u, m, moving, r, p, s, easy_cut=1, event='easy', grid=[0., .01, .02])
    assert got['raw_evidence']['equal_locality_gain_percent'] > 0
    assert not got['raw_evidence']['feasible']
    assert got['rules']['selected_risk_grid']['abstain']


def test_zero_reference_harm_forces_fallback_and_unknown_is_retained():
    u, m, moving, r, p, s = arrays()
    r[0], p[0] = 0, .00001
    r[1] = p[1] = np.nan
    got = fit_calibration(u, m, moving, r, p, s, easy_cut=1, event='all', grid=[0., .01, .02])
    assert got['rules']['selected_risk_grid']['abstain']
    assert got['raw_evidence']['selected_unknown'] == 1
    assert got['raw_evidence']['by_locality']['a']['zero_harmed'] == 1


def test_rescaling_never_expands_original_switch_set():
    u, m, moving, r, p, s = arrays()
    m[::2, 1] = .03
    p[:] = 1.05
    got = fit_calibration(u, m, moving, r, p, s, easy_cut=1, event='all', grid=[0., .01, .02])
    old = apply_calibration(u, m, moving, got['rules']['none'])
    new = apply_calibration(u, m, moving, got['rules']['population_rescale'])
    assert not (new & ~old).any()
    with pytest.raises(ValueError):
        apply_calibration(u, m, moving, dict(kind='selected_risk_grid', abstain=False, threshold=.03))


def test_inference_takes_only_causal_scores_and_frozen_rule():
    u, m, moving, r, p, s = arrays()
    got = fit_calibration(u, m, moving, r, p, s, easy_cut=1, event='all', grid=[0., .01, .02])
    first = apply_calibration(u, m, moving, got['rules']['selected_risk_grid'])
    r[:] = p[:] = np.nan
    np.testing.assert_array_equal(first, apply_calibration(u, m, moving, got['rules']['selected_risk_grid']))


def test_unsupported_event_mass_is_undefined_not_zero_risk():
    u, m, moving, r, p, s = arrays()
    e = calibration_evidence(np.zeros(8, bool), r, p, s, .1, .02)
    assert e['by_locality']['a']['positive_easy_harm_ratio'] is None
    got = fit_calibration(u, m, moving, r, p, s, easy_cut=.1, event='easy', grid=[0., .01, .02])
    assert got['rules']['population_rescale']['abstain']
