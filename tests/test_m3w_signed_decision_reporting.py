from scripts.report_m3w_moving_signed_decision import summary


def test_positive_gain_cannot_override_expansion_or_support():
    r = dict(contrasts={'logistic_minus_mse': dict(known_pp={'mean': 1.2},
        completion_lower_pp={'mean': -.2}, completion_upper_pp={'mean': 2.4})},
        expansion_supported=False)
    s = summary(r)
    assert s['known_contrast_pp'] == 1.2
    assert not s['expansion_supported'] and not s['deployment_changed']
    assert not s['completion_bounds_are_sampling_CI'] and not s['submission_ready']
    assert s['original24group_primary'] == 'not_estimable_not_passed'


def test_undefined_contrast_stays_undefined():
    r = dict(contrasts={'logistic_minus_mse': dict(known_pp={'mean': None},
        completion_lower_pp={'mean': None}, completion_upper_pp={'mean': None})},
        expansion_supported=False)
    s = summary(r)
    assert s['known_contrast_pp'] is None and s['completion_lower_pp'] is None
    assert not s['independent_confirmation']
