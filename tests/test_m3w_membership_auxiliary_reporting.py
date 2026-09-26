from copy import deepcopy
from scripts.report_m3w_european_membership_auxiliary import gates_for


def fixture():
    good=dict(positive=6,negative=0,not_estimable=0)
    metrics={k:deepcopy(good) for k in ('envelope_positive__harm_MSE_gain_percent','envelope_positive__top10_gain_pp',
        'envelope_positive__coverage_log_error_reduction','all__H_all_MSE_gain_percent')}
    return {k:{'full':deepcopy(metrics)} for k in ('aux_vs_control','aux_vs_original')}


def test_both_strong_controls_and_all_assignments_required():
    s=fixture(); assert gates_for(s)['development_cost_signal']
    for k in s:
        bad=deepcopy(s); bad[k]['full']['envelope_positive__harm_MSE_gain_percent']['positive']=5
        assert not gates_for(bad)['development_cost_signal']
    assert not gates_for(s)['new_policy_evaluated'] and not gates_for(s)['deployment_changed']


def test_tail_and_all_harm_and_missing_are_guards():
    for k in ('envelope_positive__top10_gain_pp','envelope_positive__coverage_log_error_reduction','all__H_all_MSE_gain_percent'):
        for field in ('negative','not_estimable'):
            s=fixture(); s['aux_vs_original']['full'][k][field]=1
            assert not gates_for(s)['development_cost_signal']
