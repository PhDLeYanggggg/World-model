from copy import deepcopy
from scripts.report_m3w_european_severity_auxiliary import COMPARISONS,gates_for


def summary():
    metrics={k:dict(positive=6,negative=0,not_estimable=0,overlap=0) for k in
        ('envelope_positive__harm_MSE_gain_percent','envelope_positive__top10_gain_pp',
         'envelope_positive__coverage_log_error_reduction','all__H_all_MSE_gain_percent')}
    return {k:{'full':deepcopy(metrics)} for k in COMPARISONS}


def test_each_strong_comparator_required():
    for k in COMPARISONS:
        s=summary(); s[k]['full']['envelope_positive__harm_MSE_gain_percent']['positive']=5
        assert not gates_for(s)['development_cost_signal']


def test_negative_or_missing_guard_cannot_be_hidden():
    s=summary(); assert gates_for(s)['development_cost_signal']
    for issue in ('negative','not_estimable'):
        s=summary(); s['severity_vs_original']['full']['all__H_all_MSE_gain_percent'][issue]=1
        assert not gates_for(s)['development_cost_signal']
    assert not gates_for(summary())['deployment_changed']
