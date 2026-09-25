import copy
from scripts.report_m3w_european_incremental_joint import gates, query_summary


def summary():
    m = dict(defined=36, positive_CI=1, negative_CI=0)
    return dict(queries=dict(unmatched=0, nonadditive=1, joint_unary_changed_queries=1),
        policies=dict(half_joint=dict(safety=dict(defined_views=36, worst_positive_degradation_percent=0.))),
        joint_vs_controls={p: {s: copy.deepcopy(m) for s in ('all', 'hard')} for p in ('half_independent', 'half_unary')})


def test_equal_choices_are_not_joint_lift():
    s = summary(); s['joint_vs_controls']['half_unary']['all']['positive_CI'] = 0
    assert not gates(s)['joint_beats_unary']
    assert not gates(s)['deployment_promoted']


def test_missing_roster_does_not_pass_safety_or_gain():
    s = summary(); s['policies']['half_joint']['safety']['defined_views'] = 35
    s['joint_vs_controls']['half_independent']['all']['defined'] = 35
    assert not gates(s)['easy_preservation']; assert not gates(s)['joint_beats_independent']


def test_negative_hard_and_uncertified_solver_are_retained():
    s = summary(); s['joint_vs_controls']['half_unary']['hard']['negative_CI'] = 1; s['queries']['unmatched'] = 1
    assert not gates(s)['hard_comparisons_not_negative']; assert not gates(s)['all_solvers_certified']
