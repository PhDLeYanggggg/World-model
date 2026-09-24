import copy
import numpy as np
import pytest

from scripts.verify_m3w_protected_motion_controls import check_metrics
from src.evaluation.m3w_native_metrics import paired_scene_metrics


@pytest.mark.parametrize('bootstrap', [0, 3000])
def test_separate_reducer_checks_both_point_estimate_and_scene_bootstrap(bootstrap):
    sites = np.array(['a', 'a', 'b', 'b'])
    model, ref = np.array([1., 2., 3., np.nan]), np.array([2., 3., 4., np.nan])
    report = paired_scene_metrics(model, ref, sites, expected_scenes=['a', 'b'],
        dataset='fixture', coordinate_unit='test_only', bootstrap_resamples=bootstrap)
    assert check_metrics(model, ref, sites, ['a', 'b'], report) == 2
    poisoned = copy.deepcopy(report)
    poisoned['by_scene']['a']['model_error'] += 1
    with pytest.raises(AssertionError):
        check_metrics(model, ref, sites, ['a', 'b'], poisoned)
    if bootstrap:
        poisoned = copy.deepcopy(report)
        poisoned['scene_bootstrap_ci95'][0] += 1
        with pytest.raises(AssertionError):
            check_metrics(model, ref, sites, ['a', 'b'], poisoned)


def test_zero_reference_and_absent_site_are_not_percentage_success():
    sites = np.array(['a', 'a'])
    model, ref = np.array([1., 0.]), np.zeros(2)
    report = paired_scene_metrics(model, ref, sites, expected_scenes=['a', 'b'],
        dataset='fixture', coordinate_unit='test_only', bootstrap_resamples=3000)
    assert check_metrics(model, ref, sites, ['a', 'b'], report) == 2
    assert report['equal_scene_gain_percent'] is None
