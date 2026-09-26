from copy import deepcopy
from scripts.report_m3w_european_membership_cost import gates_for


def fixture():
    good={'positive':6,'negative':0,'not_estimable':0}
    m={k:deepcopy(good) for k in ('envelope_positive__harm_MSE_gain_percent','envelope_positive__top10_gain_pp',
        'envelope_positive__coverage_log_error_reduction','all__H_all_MSE_gain_percent')}
    return {k:{'full':deepcopy(m)} for k in ('conditional_vs_direct','conditional_vs_original','conditional_vs_constant')}


def test_matched_original_and_information_control_required():
    s=fixture(); assert gates_for(s)['development_cost_signal']
    for arm in s:
        bad=deepcopy(s); bad[arm]['full']['envelope_positive__harm_MSE_gain_percent']['positive']=5
        assert not gates_for(bad)['development_cost_signal']
    assert not gates_for(s)['new_policy_evaluated']


def test_all_harm_and_missing_tail_are_guards():
    for key,field in [('all__H_all_MSE_gain_percent','negative'),('envelope_positive__top10_gain_pp','not_estimable')]:
        s=fixture(); s['conditional_vs_original']['full'][key][field]=1
        assert not gates_for(s)['development_cost_signal']
