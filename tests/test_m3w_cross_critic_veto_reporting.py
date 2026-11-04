from scripts.run_m3w_moving_cross_critic_veto import diagnostic_decision


def summary():
    return dict(primary_nonempty_matched=dict(completion_lower_pp=dict(CI95=[.1, .2])),
        matched_known_harm_reduction_pp=dict(CI95=[.1, .2]))


def test_even_a_positive_C_diagnostic_is_not_validation_or_deployment():
    result = diagnostic_decision(summary())
    assert result['worth_source_excluded_followup']
    assert result['fitting_only_not_validation_or_calibration']
    assert not result['independent_confirmation'] and not result['new_deployment'] and not result['submission_ready']


def test_empty_primary_cannot_be_replaced_by_full_strategy_secondary():
    values = summary()
    values['primary_nonempty_matched']['completion_lower_pp']['CI95'] = None
    values['secondary_full_strategy_paired'] = dict(completion_lower_pp=dict(CI95=[1., 2.]))
    assert not diagnostic_decision(values)['worth_source_excluded_followup']
