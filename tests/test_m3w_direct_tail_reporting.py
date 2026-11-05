import numpy as np

from scripts import report_m3w_moving_direct_tail_risk as report
from scripts import run_m3w_moving_direct_tail_risk as run


def test_empty_selected_reference_and_empty_easy_net_are_not_passed():
    view = dict(supported=False, selected=0, known_all_risk=None, known_easy_risk=None,
        completion_easy_risk_upper=None, known_easy_net_degradation=None)
    rows = [dict(policies={'quantile90_veto': dict(recordings={'record': view})})]
    actual = run.safety(rows, 'quantile90_veto')
    assert not actual['full_original_risk_pass']
    assert actual['supported'] == 0 and actual['undefined_selected_easy'] == 1
    assert actual['undefined_full_easy_net'] == 1
    assert actual['worst_defined_signed_easy_net_percent'] is None


def test_positive_harm_not_cancelled_by_signed_easy_net():
    view = dict(supported=False, selected=3, known_all_risk=.021, known_easy_risk=.04,
        completion_easy_risk_upper=.05, known_easy_net_degradation=-.1)
    rows = [dict(policies={'mean_veto': dict(recordings={'record': view})})]
    actual = run.safety(rows, 'mean_veto')
    assert actual['known_all_violations'] == actual['known_easy_violations'] == 1
    assert actual['worst_defined_signed_easy_net_percent'] == -10.
    assert not actual['full_original_risk_pass']


def test_reliability_keeps_unknown_outcomes_separate_and_empty_undefined():
    y = np.zeros((3, 7, 5)); y[..., 1] = 1; y[..., 2] = 10
    truth = dict(moments=y, observed_future=np.array([True, True, False]))
    p = np.full((3, 7, 2), -.1)
    a = run.reliability(p, truth, {'admitted': np.array([6, -1, 6])})['admitted']
    assert a['known_rows'] == a['unknown_rows'] == 1
    assert np.allclose(a['actual_excess_sum'], [.8, 0.])
    empty = run.reliability(p, truth, {'admitted': np.full(3, -1)})['admitted']
    assert empty['underprediction_fraction'] is None and empty['mse'] is None
    pooled = report.pool_reliability([dict(reliability={'mean': {'admitted': a}}),
        dict(reliability={'mean': {'admitted': empty}})], 'mean', 'admitted')
    assert pooled['known_rows'] == pooled['unknown_rows'] == 1
    assert pooled['underprediction_fraction'] == [1., 1.]
    assert report.number(None) == 'NA'


def test_true_upper_quantile_is_not_a_mean_harm_certificate():
    # A mathematical counterexample, not an observed dataset or a new policy.
    harm = np.array([0.]*95 + [1.]*5)
    excess = harm - .02
    assert np.quantile(excess, .9) < 0
    assert excess.mean() > 0 and harm.mean() > .02
