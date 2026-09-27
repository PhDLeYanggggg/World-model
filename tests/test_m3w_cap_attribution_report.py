from copy import deepcopy
from scripts.report_m3w_european_cap_attribution import comparisons, gates


def test_comparisons_are_fixed_and_matched():
    cs = comparisons()
    assert len(cs) == 14 and len({tuple(c) for c in cs}) == 14
    assert ['history_neighbors_envelope_coupled','old_summary_envelope_coupled'] in cs
    assert ['history_neighbors_envelope_coupled','history_neighbors_frozen_cap'] in cs


def test_joint_harm_or_missing_guard_blocks_advancement():
    record = {'primary':{'positive':2, 'negative':0, 'not_estimable':0}}
    for metric in ('top10_pp','coverage_log','all_harm'):
        record[metric] = {'positive':0, 'negative':0, 'not_estimable':0}
    summary = {'full':{a+'_vs_'+b:deepcopy(record) for a,b in comparisons()}}
    assert gates(summary)['joint_conditional_followup_justified']
    key = 'history_neighbors_envelope_coupled_vs_history_neighbors_frozen_cap'
    summary['full'][key]['all_harm']['negative'] = 1
    assert not gates(summary)['joint_conditional_followup_justified']
    summary['full'][key]['all_harm']['negative'] = 0
    summary['full'][key]['top10_pp']['not_estimable'] = 1
    assert not gates(summary)['joint_conditional_followup_justified']
    assert not gates(summary)['deployment_changed']
