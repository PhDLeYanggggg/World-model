from copy import deepcopy
from scripts.report_m3w_european_cap_auxiliary_cost import gates_for, PRIMARY, GUARDS


def fixture():
    metrics = {k: dict(positive=6, negative=0, overlap=0, not_estimable=0) for k in (PRIMARY, *GUARDS)}
    return {k: {'full': deepcopy(metrics)} for k in ('aux_vs_control', 'aux_vs_original', 'aux_vs_shuffled')}


def test_primary_requires_both_controls_and_all_assignments():
    assert gates_for(fixture())['auxiliary_cost_contribution']
    for key in ('aux_vs_control', 'aux_vs_original', 'aux_vs_shuffled'):
        s = fixture(); s[key]['full'][PRIMARY]['positive'] = 5
        assert not gates_for(s)['auxiliary_cost_contribution']


def test_negative_or_missing_guard_cannot_be_dropped():
    for key in ('aux_vs_control', 'aux_vs_original'):
        for metric in GUARDS:
            for field in ('negative', 'not_estimable'):
                s = fixture(); s[key]['full'][metric][field] = 1
                assert not gates_for(s)['auxiliary_cost_contribution']
    g = gates_for(fixture())
    assert not any(g[k] for k in ('new_policy_evaluated', 'deployment_changed', 'independent_confirmation',
                                'stage5c_executed', 'smc_enabled', 'submission_ready'))
