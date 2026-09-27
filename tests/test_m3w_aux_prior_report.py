from copy import deepcopy
from scripts.report_m3w_european_aux_prior import gates, PRIMARY, GUARDS


def summary():
    metrics = {k:dict(positive=6, negative=0, overlap=0, not_estimable=0) for k in (PRIMARY, *GUARDS)}
    return {name:dict(full=deepcopy(metrics)) for name in
        ('repair_vs_cost_only', 'repair_vs_old_true', 'repair_vs_matched_shuffled')}


def test_information_alone_never_promotes():
    s = summary(); s['repair_vs_cost_only']['full'][PRIMARY]['positive'] = 0
    assert gates(s)['true_vs_shuffled_gate'] and not gates(s)['repair_advance_gate']


def test_guard_missing_blocks_and_never_deploys():
    s = summary(); assert gates(s)['repair_advance_gate']
    assert not gates(s)['deployment_changed'] and not gates(s)['independent_confirmation']
    s['repair_vs_old_true']['full'][GUARDS[0]]['not_estimable'] = 1
    assert not gates(s)['repair_advance_gate']
