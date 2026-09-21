import numpy as np
import pytest

from src.evaluation.m3w_native_metrics import (
    native_errors, paired_scene_metrics, select_complement_baseline,
)


def summary(model, reference, sites, roster=('a', 'b'), **kwargs):
    return paired_scene_metrics(model, reference, sites, expected_scenes=roster,
                                dataset='sdd', coordinate_unit='annotation_pixel', **kwargs)


def test_mean_of_scene_percentages_not_ratio_of_scene_means():
    r = summary([.02, 100], [.01, 100], ['a', 'b'])
    assert r['equal_scene_gain_percent'] == -50
    assert r['worst_scene_gain_percent'] == -100
    assert r['by_scene']['a']['absolute_harm'] == pytest.approx(.01)


def test_scene_size_does_not_change_scene_weight():
    r = summary([0, 0, 2], [2, 2, 1], ['a', 'a', 'b'])
    assert r['equal_scene_gain_percent'] == 0


def test_zero_reference_not_fudged_or_dropped():
    r = summary([.1, .5], [0, 1], ['a', 'b'], bootstrap_resamples=3000)
    assert r['equal_scene_gain_percent'] is None
    assert r['scene_bootstrap_ci95'] is None
    assert r['by_scene']['a']['gain_percent'] is None
    assert r['by_scene']['a']['absolute_harm'] == .1


def test_missing_scene_support_keeps_roster_and_unknown():
    r = summary([.5, np.nan], [1, np.nan], ['a', 'b'])
    assert r['equal_scene_gain_percent'] is None
    assert r['by_scene']['b']['status'] == 'no_supported_labels'


@pytest.mark.parametrize('model,reference', [([np.nan, 1], [1, 1]),
    ([np.inf, 1], [1, 1]), ([-1, 1], [1, 1])])
def test_nonfinite_or_mismatched_costs_cannot_hide_rows(model, reference):
    with pytest.raises(ValueError):
        summary(model, reference, ['a', 'b'])


def test_unknown_scene_is_rejected():
    with pytest.raises(ValueError):
        summary([1, 1], [1, 1], ['a', 'unknown'])


def test_mixed_unit_pooling_rejected():
    with pytest.raises(ValueError):
        summary([1, 1], [1, 1], ['a', 'b'], row_units=['annotation_pixel', 'local'])


def test_native_restoration_and_label_mask():
    p = np.zeros((3, 12, 2)); y = p.copy(); y[:, :, 0] = 2
    mask = np.ones((3, 12), bool); mask[1, -1] = False; mask[2] = False
    y[~mask] = np.nan
    ade, fde = native_errors(p, y, mask, [.001, 100, 3])
    np.testing.assert_allclose(ade, [.002, 200, np.nan], rtol=1e-14, equal_nan=True)
    np.testing.assert_equal(fde, [.002, np.nan, np.nan])
    y[~mask] = 1e20
    np.testing.assert_equal(native_errors(p, y, mask, [.001, 100, 3]), (ade, fde))


def test_nonfinite_prediction_rejected_even_with_absent_labels():
    p = np.full((1, 12, 2), np.nan)
    with pytest.raises(ValueError):
        native_errors(p, np.zeros_like(p), np.zeros((1, 12), bool), [1])


def test_complement_selection_does_not_read_held_outcomes():
    costs = np.array([[1., 2], [1, .5], [1, .5]])
    sites = np.array(['a', 'b', 'c'])
    before, _ = select_complement_baseline(costs, sites, ('a', 'b', 'c'), reference_index=0)
    costs[0] = [1e7, 1]
    after, _ = select_complement_baseline(costs, sites, ('a', 'b', 'c'), reference_index=0)
    assert before[0] == after[0] == 1


def test_bootstrap_is_scene_level_and_reproducible():
    a = summary([.8, 1.2], [1, 1], ['a', 'b'], bootstrap_resamples=3000, seed=2)
    b = summary([.8, 1.2], [1, 1], ['a', 'b'], bootstrap_resamples=3000, seed=2)
    assert a == b
    assert a['bootstrap_unit'] == 'physical_scene'
    np.testing.assert_allclose(a['scene_bootstrap_ci95'], [-20, 20])
