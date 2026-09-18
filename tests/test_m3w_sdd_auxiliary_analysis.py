import json

import pytest

from scripts.analyze_m3w_sdd_auxiliary import event_error_summary


def test_exact_baseline_has_undefined_relative_gain_and_visible_absolute_harm():
    result = event_error_summary([{'primary_ADE': .2, 'reference_ADE': 0.}])
    assert result['gain_percent'] is None
    assert result['absolute_harm'] == .2
    json.dumps(result, allow_nan=False)


def test_event_summary_keeps_equal_scene_error_ratio():
    result = event_error_summary([{'primary_ADE': 1., 'reference_ADE': 2.},
                                  {'primary_ADE': 2., 'reference_ADE': 4.}])
    assert result['gain_percent'] == pytest.approx(50.)
    assert result['absolute_harm'] == pytest.approx(-1.5)
