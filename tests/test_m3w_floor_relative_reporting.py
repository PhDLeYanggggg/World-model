"""Aggregate reporting must retain unsupported and negative development views."""
import pytest
import numpy as np
from scripts.report_m3w_european_floor_relative import spread, policy_summary, zero_context, VARIANTS


def metric(point, lo, hi, worst=-3):
    return dict(equal_scene_gain_percent=point, scene_bootstrap_ci95=[lo, hi],
                worst_scene_gain_percent=worst)


def view(point, *, harmed=0, switch=.01, unknown=0):
    m = metric(point, point-.2, point+.2)
    return dict(ADE_vs_floor={'all': m, 'hard': m},
        ADE_vs_CV={'all': m, 'easy': metric(.1, -.1, .3, worst=-.7)},
        ADE_vs_matched_cv_target={'all': m}, FDE_vs_floor=m,
        zero_CV={'harmed_rows': harmed}, switch_rate=switch, unknown_ADE_switches=unknown)


def test_spread_retains_negative_and_crossing_intervals():
    r = spread([metric(1, .1, 2), metric(-1, -2, -.1), metric(.1, -.5, .5)])
    assert r['views'] == 3 and r['range'] == [-1, 1]
    assert r['positive_points'] == 2 and r['positive_CI'] == 1 and r['negative_CI'] == 1
    assert r['worst_locality'] == -3


def test_unknown_is_not_a_zero_error_success():
    unknown = dict(equal_scene_gain_percent=None, scene_bootstrap_ci95=None, worst_scene_gain_percent=None)
    r = spread([unknown, metric(-1, -2, -.1)])
    assert r['views'] == 2 and r['defined'] == 1 and r['positive_points'] == 0
    assert spread([unknown])['range'] is None


def test_safety_separates_positive_easy_from_zero_CV_harm():
    r = policy_summary([view(1), view(2, harmed=4, switch=.03, unknown=7)])
    assert r['safety']['worst_positive_easy_degradation_percent'] == pytest.approx(.7)
    assert r['safety']['zero_CV_harm_views'] == 1
    assert r['safety']['unknown_ADE_switches_per_view'] == [0, 7]
    assert r['safety']['switch_rate_range'] == [.01, .03]


def test_all_factorial_arms_and_old_control_are_retained():
    assert VARIANTS == ('cv_targets', 'floor_utility', 'floor_risk', 'floor_both', 'old_rebased')


def test_zero_gain_does_not_count_as_positive_transfer():
    r = spread([metric(0, 0, 0, worst=0)])
    assert r['positive_points'] == 0 and r['positive_CI'] == 0 and r['negative_CI'] == 0


def test_zero_forensics_separates_observed_motion_and_last_step_stationarity():
    h = np.zeros((3, 8, 2)); h[0, 0, 0] = 2
    valid = np.ones((3, 12), bool); valid[0, 2:] = False
    data = dict(history=h, valid=valid, sites=np.array(['a', 'b', 'b']),
                baseline_ade=np.array([[0., 0.], [0., 1.], [0., np.nan]]))
    r = zero_context(data, {0: dict(train_ids=np.array([1, 2]), held_ids=np.array([0]))})
    assert r['rows'] == 1 and r['valid_future_counts'] == [2] and r['endpoint_supported'] == 0
    assert r['last_step_stationary'] == 1 and r['accepted_by_existing_past_motion_guard'] == 1
    assert r['by_fold']['0'] == dict(fitting_zero_rows=0, held_zero_rows=1)


def test_no_zero_examples_is_not_full_horizon_safety_evidence():
    data = dict(history=np.zeros((1, 8, 2)), valid=np.ones((1, 12), bool),
                sites=np.array(['a']), baseline_ade=np.array([[0., 1.]]))
    r = zero_context(data, {})
    assert r['rows'] == 0 and r['localities'] == 0 and r['valid_future_counts'] == []
    assert r['result_source'] == 'fresh_run_posthoc_forensics_not_a_new_policy'
