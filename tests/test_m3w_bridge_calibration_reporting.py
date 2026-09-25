import pytest
from scripts.report_m3w_european_bridge_calibration import feasibility, summarize


def test_feasibility_keeps_two_probability_questions_separate_from_actual_harm():
    r = feasibility()
    assert not r['actual_harm_ratio_known_bounded_0_1']
    assert not r['confidence_level_adopted'] and not r['risk_tolerance_changed']
    twelve = [v for v in r['zero_violation_upper'] if v['n'] == 12 and v['delta'] == .05][0]
    assert .22 < twelve['bound'] < .23
    need = [v for v in r['minimum_hypothetical_zero_loss_scenes'] if v['delta'] == .05][0]
    assert need['n'] == 149


def test_reporting_rejects_partial_experiment():
    with pytest.raises(ValueError): summarize([], {})
