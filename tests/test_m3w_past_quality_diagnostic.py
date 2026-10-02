from scripts.analyze_m3w_past_quality_auxiliary import describe, risk_decomposition


def row(old_risk, new_risk, old_support=False, new_support=False):
    policies = {}
    for arm, risk, support in (('original', old_risk, old_support),
                                ('quality', new_risk, new_support),
                                ('placebo', old_risk, old_support)):
        policies[arm] = dict(easy_selected_risk_upper=risk,
                             finite_completion_supported=support,
                             selected_count=3, selected_unknown=1)
    return dict(result=dict(policies=policies, scores=dict(original=2., quality=1., placebo=3.)),
                training={a: dict(mean_leaf_training_loss_before=2.,
                                  mean_leaf_training_loss_after=1., training_zero_mean_max=1e-15)
                          for a in ('quality', 'placebo')},
                diagnostic={a: dict(negative_raw_coordinates=2) for a in ('quality', 'placebo')})


def test_undefined_is_not_resolved_risk_violation():
    x = describe([row(.03, None)])
    assert x['transitions']['quality']['old_upper_violations_now_undefined_not_passes'] == 1
    assert x['transitions']['quality']['old_upper_violations_now_defined_below_budget'] == 0
    assert x['arms']['quality']['worst_easy_risk_upper'] is None


def test_preserves_separate_risk_and_support_transitions():
    x = describe([row(.03, .01, False, True), row(.01, .04, True, False)])
    t = x['transitions']['quality']
    assert t['complete_support_lost'] == t['complete_support_gained'] == 1
    assert t['new_upper_violations'] == t['old_upper_violations_now_defined_below_budget'] == 1
    assert x['arms']['quality']['validation_MSE_better_than_original_groups'] == 2


def test_counts_unknown_and_repeated_occurrences_without_dropping():
    x = describe([row(None, None), row(None, None)])
    assert x['arms']['quality']['unknown_selected_occurrences'] == 2
    assert x['arms']['quality']['undefined_easy_risk'] == 2
    assert x['arms']['quality']['selected_occurrences'] == 6


def test_completion_upper_is_not_mislabeled_observed_harm():
    r = row(.01, 10.)
    r.update(group='fixed', head_seed=17)
    for name, p in r['result']['policies'].items():
        p.update(selected_known_easy_reference_mass=2.,
                 selected_known_easy_harm_mass=.02,
                 selected_unknown_envelope_mass=19.98 if name == 'quality' else 0.,
                 easy_degradation_upper=0.)
    d = risk_decomposition([r])
    assert d['quality']['known_label_violations'] == 0
    assert d['quality']['completion_only_violations'] == 1
    assert d['quality']['worst_upper_case']['known_easy_risk'] == .01
