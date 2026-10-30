from scripts import report_m3w_admission_monitor as report


def test_complete_exact_replay_roster_is_training_evidence_not_new_policy_scores():
    complete, summary, rows = report.build()
    assert len(rows) == 19 and complete['actual_optimizer_updates'] == 38000
    assert summary['admitted_training_totals']['step'] == 36000
    assert summary['admitted_training_totals']['unknown_supervised_draws'] == 0
    assert all(r['scientific_state_exact'] for r in rows)
    assert summary['no_new_candidate_or_policy']
    assert not summary['effectiveness_recomputed'] and not summary['independent_confirmation']


def test_measured_activation_and_weak_gradients_are_not_conflated():
    _, summary, _ = report.build()
    assert summary['admitted_training_totals']['steps_aux_positive'] == 33771
    assert min(summary['first_aux_positive_steps']) == 22
    assert max(summary['first_aux_positive_steps']) == 90
    ratios = summary['snapshot_gradient_distributions']
    assert ratios['auxiliary_over_anchor_gradient']['quantiles'][2] < .0001
    assert ratios['auxiliary_over_anchor_gradient']['defined'] == 738
    assert not summary['deployment_changed'] and not summary['stage5c_executed'] and not summary['smc_enabled']
