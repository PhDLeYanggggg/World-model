from copy import deepcopy
from scripts.report_m3w_moving_record_excess_cost import scientific_decision


def summary():
    effect = dict(CI95=[.1, .2])
    return dict(paired={a: dict(joint_free=dict(completion_lower_pp=effect)) for a in ('point', 'shared_tree')},
        completion_lower=dict(record_excess=dict(joint_free=effect)),
        safety=dict(record_excess=dict(joint_free=dict(required_record_views=348, supported_record_views=348,
            full_original_risk_pass=True, undefined_easy_views=0, known_all_risk_violations=0,
            known_easy_risk_violations=0, worst_known_easy_net_degradation_percent=1.))))


def test_easy_net_and_positive_mean_do_not_replace_support():
    s = summary()
    s['safety']['record_excess']['joint_free']['undefined_easy_views'] = 1
    assert not scientific_decision(s)['supported_source_mechanism_gate']


def test_missing_primary_or_strong_control_cannot_pass():
    s = summary()
    s['paired']['shared_tree']['joint_free']['completion_lower_pp'] = dict(CI95=None)
    assert not scientific_decision(s)['supported_source_mechanism_gate']


def test_development_mechanism_never_promotes_independent_deployment():
    d = scientific_decision(deepcopy(summary()))
    assert d['supported_source_mechanism_gate']
    assert not d['independent_confirmation'] and not d['new_deployment'] and not d['submission_ready']
