from copy import deepcopy
from scripts.report_m3w_moving_adaptive_record_cost import scientific_decision


def summary():
    effect = dict(CI95=[.1, .2])
    return dict(paired={a: dict(joint_free=dict(completion_lower_pp=effect))
        for a in ('point', 'shared_tree', 'frozen_teacher')},
        completion_lower=dict(adaptive=dict(joint_free=effect)),
        safety=dict(adaptive=dict(joint_free=dict(required_record_views=348, supported_record_views=348,
            full_original_risk_pass=True, undefined_easy_views=0, known_all_risk_violations=0,
            known_easy_risk_violations=0, worst_known_easy_net_degradation_percent=1.))))


def test_all_required_record_support_cannot_be_replaced_by_easy_net():
    s = summary()
    s['safety']['adaptive']['joint_free']['supported_record_views'] = 347
    assert not scientific_decision(s)['supported_source_mechanism_gate']


def test_primary_or_strong_control_missing_or_nonfinite_cannot_pass():
    for control in ('shared_tree', 'frozen_teacher'):
        for value in (None, [float('nan'), .2], [-.1, .3]):
            s = summary()
            s['paired'][control]['joint_free']['completion_lower_pp'] = dict(CI95=value)
            assert not scientific_decision(s)['supported_source_mechanism_gate']


def test_positive_development_contrast_does_not_promote_deployment():
    decision = scientific_decision(deepcopy(summary()))
    assert decision['supported_source_mechanism_gate']
    assert not decision['new_deployment'] and not decision['submission_ready']
    assert not decision['independent_confirmation'] and not decision['new_trajectory_training']
