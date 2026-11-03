import numpy as np
import pytest

from src.evaluation.m3w_eqmotion_amplitude import attenuate, risk_diagnosis, locality_interval
from src.evaluation.m3w_selection_forecast_readout import errors, policy


def test_attenuation_endpoints_and_no_row_pruning():
    b = np.zeros((3, 12, 2)); p = np.ones_like(b); p[0] = 0
    np.testing.assert_array_equal(attenuate(b, p, 0), b)
    np.testing.assert_array_equal(attenuate(b, p, 1), p)
    np.testing.assert_array_equal(attenuate(b, p, .25), .25*p)
    for alpha in (.001, .01, .25, 1):
        np.testing.assert_array_equal(np.any(attenuate(b, p, alpha) != b, axis=(1, 2)),
                                      np.any(p != b, axis=(1, 2)))
    for alpha in (-1, 2, np.nan):
        with pytest.raises(ValueError): attenuate(b, p, alpha)


def test_unknown_radius_is_scaled_not_a_zero_error_and_absent_endpoint_is_not_used():
    b = np.zeros((3, 12, 2)); p = np.ones_like(b); y = np.full_like(b, 2)
    valid = np.ones((3, 12), bool); valid[1, 6:] = False; valid[2] = False
    be, _ = errors(b, y, valid); rec = np.array([1, 1, 1]); easy = np.array([True, True, False])
    a = risk_diagnosis(attenuate(b, p, .25), b, y, valid, rec, easy=easy, baseline_cost=be)['1']
    full = risk_diagnosis(p, b, y, valid, rec, easy=easy, baseline_cost=be)['1']
    assert a['unknown_disagreement_radius'] == full['unknown_disagreement_radius']*.25
    assert a['completion_all_risk_upper'] > a['known_all_risk']
    y[~valid] = 1e30
    assert risk_diagnosis(attenuate(b, p, .25), b, y, valid, rec, easy=easy, baseline_cost=be)['1'] == a


def test_zero_intervention_and_zero_reference_are_not_supported_success():
    b = np.zeros((2, 12, 2)); y = np.ones_like(b); valid = np.ones((2, 12), bool)
    result = policy(b, b, b, y, valid, np.array([1, 2]), easy_cut=4, hard_cut=5)
    assert all(not v['supported'] and v['completion_easy_risk_upper'] is None
               for v in result['recordings'].values())


def test_interval_preserves_complete_roster_and_missing_denominators():
    rows = [dict(site=s, trial=f'{f}_{seed}', seed=seed, value=1.)
            for s in ('a', 'b') for f in (0, 1) for seed in (17, 29, 43)]
    get = lambda r: r['value']
    assert locality_interval(rows, get, ['a', 'b'], 100, 17)['CI95'] == [1., 1.]
    with pytest.raises(ValueError): locality_interval(rows[:-1], get, ['a', 'b'], 100, 17)
    rows[0]['value'] = None
    assert locality_interval(rows, get, ['a', 'b'], 100, 17)['point'] is None
