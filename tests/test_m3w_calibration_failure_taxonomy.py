import pytest
from src.evaluation import m3w_calibration_failure_taxonomy as api


def view(**kw):
    out = dict(known_all_risk=.01, known_easy_risk=.01,
        completion_all_risk_upper=.01, completion_easy_risk_upper=.01,
        known_easy_net_degradation=.01, selected=8, unknown_selected=0, supported=True)
    out.update(kw)
    return out


def test_exclusive_and_overlapping_failure_causes():
    cases = [
        (view(), 'supported'),
        (view(known_easy_risk=None, completion_easy_risk_upper=None, supported=False), 'unsupported_reference'),
        (view(known_all_risk=.03, completion_all_risk_upper=.05, supported=False), 'observed_positive_harm'),
        (view(known_easy_net_degradation=.03, supported=False), 'easy_net_only'),
        (view(completion_easy_risk_upper=.08, unknown_selected=1, supported=False), 'unknown_completion_bound_only')]
    for data, category in cases:
        assert api.classify(data)['category'] == category
    both = api.classify(view(known_all_risk=.2, known_easy_risk=None,
                            completion_easy_risk_upper=None, supported=False))
    assert both['category'] == 'unsupported_reference'
    assert both['observed_positive_harm_violation']


def test_exact_budget_passes_but_undefined_empty_support_does_not():
    assert api.classify(view(known_all_risk=.02, completion_all_risk_upper=.02))['category'] == 'supported'
    empty = view(known_all_risk=None, known_easy_risk=None, completion_all_risk_upper=None,
                 completion_easy_risk_upper=None, selected=0, supported=False)
    assert api.classify(empty)['category'] == 'unsupported_reference'
    assert api.classify(empty)['selected_none']
    assert not api.classify(empty)['full_roster_easy_reference_unavailable']
    full = dict(empty, known_easy_net_degradation=None)
    assert api.classify(full)['full_roster_easy_reference_unavailable']


def test_bad_flags_and_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        api.classify(view(known_all_risk=.3))
    with pytest.raises(ValueError):
        api.classify(view(known_all_risk=float('nan'), supported=False))


def test_compatibility_with_actual_partial_label_metric_interface():
    import numpy as np
    from src.evaluation.m3w_selection_forecast_readout import policy
    floor = np.zeros((4, 12, 2)); floor[..., 0] = 1.
    predicted = floor.copy(); predicted[:2, :, 0] += .005
    target = np.zeros_like(floor); valid = np.ones((4, 12), bool); valid[-1] = False
    def cause(p):
        score = policy(p, floor, floor, target, valid, np.ones(4, int), easy_cut=2., hard_cut=10.)
        return api.classify(next(iter(score['recordings'].values())))
    assert cause(predicted)['category'] == 'supported'
    predicted[-1, :, 0] += .05
    assert cause(predicted)['category'] == 'unknown_completion_bound_only'
    predicted[0, :, 0] += .5
    assert cause(predicted)['category'] == 'observed_positive_harm'
