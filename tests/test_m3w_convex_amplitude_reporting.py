import json
import numpy as np
import pytest
from scripts import report_m3w_moving_convex_amplitude as report


def test_source_macro_usage_and_empty_matched_views_stay_distinct():
    rows=[]
    for site,values in [('a',[0.,1.]),('b',[1.])]:
        for value in values:
            r=dict(site=site,policies={},amplitudes={},matched_utility={})
            for arm in ('unconstrained','convex'):
                r['policies'][arm]=dict(joint_free=dict(switch_rate=value))
                r['amplitudes'][arm]=dict(joint_free=dict(mean_amplitude_all_rows=value/2,
                    action_counts=[1,0,1,0,0,0,0,0]))
                r['matched_utility'][arm]=dict(contrast=dict(known_pp=None if value==0 else 0.))
            rows.append(r)
    result=report.usage(rows)
    assert all(r['source_macro_switch_percent']==75. for r in result)
    assert all(r['source_macro_amplitude_all_rows']==.375 for r in result)
    assert all(r['empty_matched_group_views']==1 for r in result)
    assert all(not r['full_scope_same_count_utility_estimable']for r in result)


def test_both_original_budget_solver_ledgers_are_required(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    private=tmp_path/'new';monkeypatch.setattr(report,'PRIVATE',private)
    original=tmp_path/'data/stage_cvpr2027_experiments/moving_amplitude_cost_v1'
    jobs=[f'job{i}'for i in range(18)]
    meta=dict(bindings=dict(seconds=.25),completed_queries=3,
        reason_counts={p:dict(exact_milp=3)for p in ('joint_matched','joint_free')},
        nonoptimal_queries=dict(joint_free=0),changed_queries=dict(joint_free=1))
    for root,tag in ((private,'convex'),(original,'amplitude_bank')):
        (root/'allocation').mkdir(parents=True)
        for job in jobs:
            np.savez_compressed(root/'allocation'/(job+'_'+tag+'.npz'),metadata_json=np.asarray(json.dumps(meta)))
    rows=report.solvers(jobs)
    assert len(rows)==36
    assert sum(r['result_source']=='cached_verified'for r in rows)==18
    meta['bindings']['seconds']=.5
    np.savez_compressed(private/'allocation/job0_convex.npz',metadata_json=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError,match='solver-budget'):
        report.solvers(jobs)
