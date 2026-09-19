import numpy as np
import pytest

from src.evaluation.m3w_source_population import (
    baseline_errors, scene_mean, contribution, complement_selected_baseline,
    window_support, baseline_table, scale_floor_mask,
)


def test_legacy_float32_floor_in_float64_container():
    np.testing.assert_array_equal(scale_floor_mask([.001, float(np.float32(.001)), .002]), [True, True, False])


def test_missing_future_is_unknown_not_stationary_negative():
    g = np.zeros((3, 476)); y = np.zeros((3, 12, 2)); m = np.ones((3, 12), bool)
    m[1, -1] = False; m[2] = False; y[~m] = np.nan
    ade, fde, events = baseline_errors(g, y, m)
    np.testing.assert_array_equal(events, [0, -1, -1])
    assert np.isnan(ade[2]).all() and np.isnan(fde[1:]).all()
    assert (ade[:2] == 0).all()


def test_static_start_scale_dominance_is_not_headroom():
    g = np.zeros((2, 476)); y = np.zeros((2, 12, 2)); y[0, :, 0] = 1000
    m = np.ones((2, 12), bool)
    ade, fde, e = baseline_errors(g, y, m)
    np.testing.assert_array_equal(e, [1, 0])
    table = baseline_table(ade, fde, ['site', 'site'], [.001, .001])
    assert table['oracle_headroom_over_cv_percent'] == 0
    assert table['baseline_metrics']['constant_position']['native_pixel_ade_diagnostic'] == .5


def test_invalid_future_payload_does_not_change_supported_error():
    rng = np.random.default_rng(1)
    g = np.zeros((4, 476)); y = rng.normal(size=(4, 12, 2)); m = np.ones((4, 12), bool)
    m[:, -3:] = False
    first = baseline_errors(g, y, m)
    y[~m] = 1e25
    second = baseline_errors(g, y, m)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('column', [16, 166])
def test_future_input_time_rejected(column):
    g = np.zeros((1, 476)); g[0, column] = 1; g[0, 230] = 1
    with pytest.raises(ValueError):
        baseline_errors(g, np.zeros((1, 12, 2)), np.ones((1, 12), bool))


def test_equal_site_contribution_does_not_reweight_small_subset():
    values = np.array([100., 0., 2.]); sites = np.array(['a', 'a', 'b'])
    assert scene_mean(values, sites) == 26
    result = contribution(values, [True, False, False], sites)
    assert result['equal_site_error_contribution'] == 25
    assert result['percent_of_total'] == pytest.approx(100*25/26)


def test_complement_selection_cannot_use_held_costs():
    sites = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    costs = np.ones((6, 7)); costs[2:, 3] = .5
    before = complement_selected_baseline(costs, sites)['folds']['a']['selected']
    costs[:2, :] = np.arange(7)*100
    after = complement_selected_baseline(costs, sites)['folds']['a']['selected']
    assert before == after == 'damped_velocity_010'


def test_scoped_tracks_and_overlapping_spans_are_not_independent():
    report = window_support(['a:1', 'a:1', 'a:1', 'b:1'], [0, 12, 240, 0], [0, 0, 1, 0])
    assert report['tracks'] == 2 and report['disjoint_spans'] == 3
    assert report['annotation_category_runs'] == 3
    assert report['independent_events_claimed'] is False
