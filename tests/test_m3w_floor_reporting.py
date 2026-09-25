import pytest
from scripts.report_m3w_european_floor_opportunity import distribution, dist_metrics


def test_missing_locality_views_stay_in_denominator():
    result = distribution([dict(equal_locality=None, ci95=None),
                           dict(equal_locality=1., ci95=[-.5, 2.])])
    assert result['views'] == 2 and result['defined'] == 1
    assert result['positive_points'] == 1 and result['positive_CI'] == 0


def test_all_undefined_is_not_zero_gain_or_pass():
    result = distribution([dict(equal_locality=None, ci95=None)])
    assert result['range'] is None and result['defined'] == 0
    assert result['positive_CI'] == result['negative_CI'] == 0


def test_summary_keeps_negative_and_uncertain_groups():
    result = dist_metrics([
        dict(equal_scene_gain_percent=-2., scene_bootstrap_ci95=[-3., -1.]),
        dict(equal_scene_gain_percent=1., scene_bootstrap_ci95=[-.2, 2.]),
        dict(equal_scene_gain_percent=.5, scene_bootstrap_ci95=[.1, .7])])
    assert result['range'] == pytest.approx([-2, 1])
    assert result['positive_points'] == 2
    assert result['positive_CI'] == 1 and result['negative_CI'] == 1
