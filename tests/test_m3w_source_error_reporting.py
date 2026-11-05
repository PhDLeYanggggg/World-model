from scripts.report_m3w_moving_source_error_priority import interpretation


def test_completion_bounds_and_positive_development_screen_are_not_safety():
    raw = dict(contrasts=dict(held_minus_in_source={k: dict(mean=v) for k, v in
        [('known_pp', 1), ('completion_lower_pp', .1), ('completion_upper_pp', 2)]}),
        expansion_supported=True)
    result = interpretation(raw)
    assert result['expansion_supported']
    assert not result['completion_bounds_are_sampling_CI']
    assert not result['deployment_changed'] and not result['submission_ready']
    assert not result['independent_confirmation']
    assert result['original24group_primary'] == 'not_estimable_not_passed'
    raw['expansion_supported'] = False
    raw['contrasts']['held_minus_in_source']['known_pp']['mean'] = -.5
    assert interpretation(raw)['known_contrast_pp'] == -.5
    assert not interpretation(raw)['expansion_supported']
