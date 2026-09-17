import numpy as np

from scripts.analyze_m3w_appearance_support_control import decompose


def test_unchanged_predictions_separate_gate_effect():
    p = np.ones((3, 12, 2)) * 5
    target = np.ones_like(p)
    result = decompose(p, p.copy(), target, np.ones(3), np.ones(3, bool), np.zeros(3, bool))
    assert result['prediction_component_error_change'] == 0
    assert result['gate_component_error_change'] < 0
    assert result['actual_gain_pct'] == 0
    assert result['treated_switch_count'] == 0


def test_unchanged_gate_separates_forecast_effect():
    target = np.ones((3, 12, 2))
    result = decompose(target * 5, target, target, np.ones(3), np.ones(3, bool), np.ones(3, bool))
    assert result['gate_component_error_change'] == 0
    assert result['prediction_component_error_change'] < 0
    assert result['actual_gain_pct'] == 100
    assert result['original_switch_count'] == result['treated_switch_count'] == 3
