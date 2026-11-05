from scripts.report_m3w_moving_annealed_risk import summary


def test_training_or_average_gain_does_not_override_failed_safety():
    a = dict(known={'mean': 3.6}, hard={'mean': 4.6},
        safety={'supported': 0, 'required': 17, 'worst_defined_signed_easy_net_percent': 38.9})
    b = dict(known={'mean': 10.}, hard={'mean': 20.},
        safety={'supported': 16, 'required': 17, 'worst_defined_signed_easy_net_percent': 2.1})
    r = summary(dict(held_policy={'constant_veto': a, 'cosine_veto': b},
        contrasts={'cosine_minus_constant': {'known_pp': {'mean': 6.4}}},
        expansion_supported=True, original24group_primary='not_estimable_not_passed'))
    assert r['known_gain_after'] > r['known_gain_before']
    assert not r['full_recording_risk_support_pass']
    assert not r['deployment_changed'] and not r['submission_ready']
    assert not r['completion_bounds_are_sampling_CI']
    assert r['original24group_primary'] == 'not_estimable_not_passed'
