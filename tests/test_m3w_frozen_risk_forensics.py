from copy import deepcopy

import pytest

from src.evaluation.m3w_frozen_risk_forensics import audit_query, summarize_queries


def rows_and_query(values, *, predicted_harm=0.001, budget=0.01):
    rows=[]
    for i,(baseline,selected,switch) in enumerate(values):
        rows.append(dict(recording_id='record',physical_scene='site',frame_id=10,
                         horizon_raw=120,agent_id=i,baseline_ade=baseline,
                         arms={'control':dict(ade=selected,switch=switch)}))
    query=dict(recording_id='record',physical_scene='site',frame_id=10,horizon_raw=120,
               agent_count=len(rows),arms={'control':dict(mean_predicted_harm=predicted_harm,
                                                        predicted_constraints_satisfied=True)})
    return rows,query,dict(control='control',query_control='control',budget=budget,easy_threshold=.02)


def test_truthful_overall_budget_does_not_protect_easy_ratio():
    values=[(.001,.1,True)]+[(1.,1.,False)]*99
    rows,q,kwargs=rows_and_query(values,predicted_harm=.099/100)
    r=audit_query(rows,q,**kwargs)
    assert r['budget_status']=='known_within'
    assert r['harm_lower_bound']==pytest.approx(r['predicted_mean_harm'])
    s=summarize_queries([r],[0.,.25,.5,.75,1.00000001])
    assert s['easy_degradation_percent']==pytest.approx(9900.)
    assert s['easy_positive_harm_fraction']['known_within']==1.


def test_missing_selected_labels_never_count_as_zero_risk():
    rows,q,kwargs=rows_and_query([(None,None,True)])
    r=audit_query(rows,q,**kwargs)
    assert r['budget_status']=='indeterminate'
    assert r['harm_lower_bound']==0. and r['realized_harm_exact'] is None
    s=summarize_queries([r],[0.,.25,.5,.75,1.00000001])
    assert s['easy_degradation_percent'] is None
    assert s['unknown_selected_agents']==1
    assert s['reliability_bins'][0]['exact_harm_mean'] is None


def test_lower_bound_can_prove_failure_despite_missing_labels():
    rows,q,kwargs=rows_and_query([(.001,.101,True),(None,None,True)])
    r=audit_query(rows,q,**kwargs)
    assert r['harm_lower_bound']==pytest.approx(.05)
    assert r['budget_status']=='proven_exceeds'
    assert r['realized_harm_exact'] is None


def test_unselected_missing_labels_have_structural_zero_excess():
    rows,q,kwargs=rows_and_query([(None,None,False),(.01,.01,False)])
    r=audit_query(rows,q,**kwargs)
    assert r['budget_status']=='known_within'
    assert r['realized_harm_exact']==0.
    assert r['missing_ade_agents']==1 and r['unknown_selected_agents']==0


def test_benefits_do_not_cancel_positive_harm_budget():
    rows,q,kwargs=rows_and_query([(1.,6.,True),(6.,1.,True)])
    r=audit_query(rows,q,**kwargs)
    assert r['observed_net_excess_sum']==0.
    assert r['harm_lower_bound']==2.5 and r['budget_status']=='proven_exceeds'


def test_reject_identity_population_mask_and_fake_fallback():
    rows,q,kwargs=rows_and_query([(.01,.02,True),(.01,.01,False)])
    for changed in (rows+rows[:1],rows[:1]):
        with pytest.raises(ValueError):audit_query(changed,q,**kwargs)
    changed=deepcopy(rows);changed[1]['agent_id']=0
    with pytest.raises(ValueError,match='identity'):audit_query(changed,q,**kwargs)
    changed=deepcopy(rows);changed[0]['frame_id']=9
    with pytest.raises(ValueError,match='identity'):audit_query(changed,q,**kwargs)
    changed=deepcopy(rows);changed[0]['baseline_ade']=None
    with pytest.raises(ValueError,match='label'):audit_query(changed,q,**kwargs)
    changed=deepcopy(rows);changed[1]['arms']['control']['ade']=.5
    with pytest.raises(ValueError,match='fallback'):audit_query(changed,q,**kwargs)


def test_query_risk_and_agent_aggregate_denominators_are_distinct():
    a,q,k=rows_and_query([(.01,.011,True)]);one=audit_query(a,q,**k)
    a,q,k=rows_and_query([(.01,.012,True)]*9);two=audit_query(a,q,**k)
    two['frame_id']=11
    r=summarize_queries([one,two],[0.,.25,.5,.75,1.00000001])
    assert r['mean_query_harm_lower_bound']==pytest.approx(.0015)
    assert r['mean_past_agent_harm_lower_bound']==pytest.approx(.0019)
    assert r['easy_degradation_percent']==pytest.approx(19.)
    with pytest.raises(ValueError,match='duplicated'):summarize_queries([one,one],[0.,1.])


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf'),True])
def test_invalid_observed_errors_are_rejected(bad):
    rows,q,kwargs=rows_and_query([(.01,.02,True)])
    rows[0]['arms']['control']['ade']=bad
    with pytest.raises(ValueError,match='nonnegative'):
        audit_query(rows,q,**kwargs)


def test_predicted_constraint_claim_and_fixed_bins_cannot_hide_failures():
    rows,q,kwargs=rows_and_query([(.01,.02,True)],predicted_harm=.02)
    with pytest.raises(ValueError,match='contradicts'):
        audit_query(rows,q,**kwargs)
    q['arms']['control']['predicted_constraints_satisfied']=False
    r=audit_query(rows,q,**kwargs)
    with pytest.raises(ValueError,match='outside fixed bins'):
        summarize_queries([r],[0.,1.])
    summary=summarize_queries([r],[0.,1.,2.])
    assert summary['predicted_budget_failure_queries']==1
    assert summary['reliability_bins'][-1]['queries']==1
