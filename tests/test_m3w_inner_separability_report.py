import numpy as np
import pytest
from scripts import report_m3w_inner_separability as report


def test_undefined_is_not_zero_or_complete():
    m = report.complete_mean([1, None, 3])
    assert m['mean'] is None and m['conditional_mean'] == 2
    assert m['undefined'] == 1


def test_bootstrap_collapses_repeated_views_first():
    rows = [dict(site='a', value=1)]*100+[dict(site='b', value=3)]
    result = report.locality_interval(rows, 'value')
    assert result['summary']['mean'] == 2
    assert result == report.locality_interval(rows, 'value')


def test_missing_locality_view_blocks_full_interval():
    result = report.locality_interval([dict(site='a', x=None), dict(site='b', x=0)], 'x')
    assert result['CI95'] is None


def test_risk_no_switch_is_not_pass():
    row = dict(metric=dict(risk=None, unknown_interventions=0))
    result = report.risk_coverage([row], 'risk')
    assert result['undefined'] == 1 and not result['all_views_pass']


def test_unknown_intervention_is_not_safety_pass():
    row = dict(metric=dict(risk=0., unknown_interventions=1))
    assert not report.risk_coverage([row], 'risk')['all_views_pass']


def test_independent_metric_check_detects_corruption():
    from scripts.run_m3w_european_fixed_floor_probe import metric
    cv = np.array([1., 2., np.nan]); floor = cv.copy()
    neural = np.array([.5, 3., np.nan]); take = np.array([True, True, False])
    valid = np.array([[True, True], [True, True], [False, False]])
    m = metric(cv, floor, neural, cv, floor, neural, np.where(take, neural, floor),
               np.where(take, neural, floor), take, valid, 1., 2.)
    m['selected_easy_positive_harm_ratio'] = 0.
    assert report.audit_metric(m, cv, floor, neural, cv, floor, neural, take, valid, 1., 2.) > 20
    m['error_sum'] += 1
    with pytest.raises(AssertionError):
        report.audit_metric(m, cv, floor, neural, cv, floor, neural, take, valid, 1., 2.)
