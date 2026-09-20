from copy import deepcopy

import numpy as np
import pytest

from src.evaluation.m3w_development_evaluation import score_scene
from src.evaluation.m3w_interaction_control_evaluation import attach_interaction_controls
from src.evaluation.m3w_frozen_interaction import verify_original_rows,compact_query,summarize_controls,CONTROLS


def fixture():
    b = np.array([[[0.,0.]],[[3.,0.]],[[9.,0.]]])
    c = np.array([[[1.5,0.]],[[1.5,0.]],[[9.,1.]]])
    ids = [1,2,3]
    scene = dict(recording_id='fixture',physical_scene='one_site',frame_id=0,horizon_raw=1,
        agents=[dict(agent_id=i,inputs={'prediction_frame_offsets':np.array([1.])},
            coordinate_transform={'origin_xy':np.zeros(2),'rotation':np.eye(2),'scale':1.}) for i in ids])
    arm = lambda pred:dict(prediction=pred,switch=np.zeros(3,bool),mean_pair_proxy=0.,reason='fixture',
                          predicted_constraints_satisfied=True)
    decision = dict(agent_ids=ids,baseline=b,candidate=c,predicted_gain=np.array([1.,.9,.8]),
        predicted_harm=np.zeros(3),supported=np.ones(3,bool),
        arms={k:arm(c if k=='uncontrolled' else b) for k in ('floor','uncontrolled','independent','scene_uniform','joint')})
    policy = dict(pair_weight=1.,max_mean_predicted_harm=.1,max_intervention_fraction=2/3,
                  min_predicted_gain=0.,max_agent_predicted_harm=1.)
    result = attach_interaction_controls(scene,decision,policy=policy,geometry={'graph_radius':20.,'proximity_threshold':.75})
    labels = [dict(agent_id=i,future_frame_ids=np.array([1.]),future_xy_dataset_local=np.ones((1,2)),
                   future_label_mask=np.array([i!=2])) for i in ids]
    original = score_scene(scene,decision,labels,label_policy='complete_requested_path')
    new = score_scene(scene,result,labels,label_policy='complete_requested_path')
    query = compact_query(scene,result)
    protocol = dict(task={'primary_metric':'ade','aggregation':'equal_physical_scene'},bootstrap_resamples=2000,
        development_evaluation=dict(bootstrap_seed=3,error_unit='past_normalized',easy_threshold=.1,hard_threshold=1.))
    return original,new,query,protocol


def test_exact_forecast_comparison_rejects_identity_mask_or_forecast_changes():
    original,new,_,_ = fixture()
    assert verify_original_rows(original,new)['agent_rows_verified'] == 3
    for key,value in [('scale',2.),('agent_id',99),('baseline_ade',None),('available_label_steps',0)]:
        changed = deepcopy(new);changed[0][key] = value
        with pytest.raises(ValueError,match='query, scale or label'):
            verify_original_rows(original,changed)
    changed = deepcopy(new);changed[0]['arms']['uncontrolled']['ade'] += 1
    with pytest.raises(ValueError,match='forecast scoring'):
        verify_original_rows(original,changed)
    with pytest.raises(ValueError,match='population'):
        verify_original_rows(original,new[:-1])


def test_legacy_control_differences_are_disclosed_not_forecast_repair():
    original,new,_,_ = fixture()
    new[0]['arms']['joint']['switch'] = True
    result = verify_original_rows(original,new)
    assert result['legacy_control_rows_different'] == 1
    assert result['fixed_forecast_error_rows_different'] == 0


def test_single_scene_cannot_claim_bootstrap_and_missing_labels_keep_counts():
    _,rows,query,protocol = fixture()
    r = summarize_controls(rows,[query],protocol)
    assert r['query_summary']['agent_queries'] == 3
    assert r['query_summary']['matched_queries'] == 1
    assert r['full']['label_coverage'] == {'ade':2,'fde':2}
    assert r['full']['arms'][CONTROLS[2]]['all']['bootstrap']['status'] == 'not_run_insufficient_physical_scenes'
    contrast = r['contrasts']['matched_nonzero'][CONTROLS[2]+'_minus_'+CONTROLS[1]]['ade']
    assert contrast['count'] == 2 and contrast['ci95'] is None
    assert contrast['selected_past_agent_count_matched']
    assert contrast['scored_agent_count_matched'] is None and not contrast['realized_risk_matched']
    assert not r['eligible_for_selection']


def test_unmatched_queries_remain_in_full_accuracy_summary():
    _,rows,query,protocol = fixture()
    query.update(matched=False,nonzero_matched=False)
    r = summarize_controls(rows,[query],protocol)
    assert r['query_summary']['unmatched_queries'] == 1
    assert r['full']['arms'][CONTROLS[2]]['all']['count'] == 2
    for pair in r['contrasts']['matched'].values():
        assert pair['ade']['status'] == 'not_run_no_eligible_labels'


def test_independent_cost_reduction_rejects_false_choices_or_count_claims():
    from scripts.analyze_m3w_frozen_interaction import independent_check
    _,rows,query,protocol = fixture()
    summary = summarize_controls(rows,[query],protocol)
    policy = {'pair_weight':1.}
    result = independent_check(rows,[query],summary,policy)
    assert result['selected_forecast_error_checks'] == 18
    changed = deepcopy(rows);changed[0]['arms'][CONTROLS[2]]['ade'] += 1
    with pytest.raises(ValueError,match='fixed floor/candidate'):
        independent_check(changed,[query],summary,policy)
    changed_query = deepcopy(query);changed_query['reference_count'] += 1
    with pytest.raises(ValueError,match='count match'):
        independent_check(rows,[changed_query],summary,policy)
    changed_summary = deepcopy(summary)
    changed_summary['full']['arms'][CONTROLS[1]]['all']['selected_error'] += 1
    with pytest.raises(AssertionError):
        independent_check(rows,[query],changed_summary,policy)
    with pytest.raises(ValueError,match='Duplicate agent query'):
        independent_check(rows+rows[:1],[query],summary,policy)
    with pytest.raises(ValueError,match='membership duplicated or incomplete'):
        independent_check(rows,[query,query],summary,policy)
    with pytest.raises(ValueError,match='membership duplicated or incomplete'):
        independent_check(rows,[],summary,policy)
    changed_summary = deepcopy(summary)
    key = CONTROLS[2]+'_minus_'+CONTROLS[1]
    changed_summary['contrasts']['all'][key]['ade']['left_minus_right_error'] += 1
    with pytest.raises(AssertionError):
        independent_check(rows,[query],changed_summary,policy)
    changed_query=deepcopy(query)
    changed_query['arms']['joint']['full_objective']+=.001
    changed_query['arms']['joint']['mean_predicted_gain']-=.001
    with pytest.raises(ValueError,match='float32 reduction bound'):
        independent_check(rows,[changed_query],summary,policy)


def test_float32_mean_display_is_not_a_float64_solver_certificate():
    from scripts.analyze_m3w_frozen_interaction import display_roundoff_bound
    values=np.array([.132317,.614,.782196,.042911,.134593,.029878],dtype=np.float32)
    display=float(np.mean(values))
    coefficients=float(np.sum((values/len(values)).astype(np.float64)))
    assert np.mean(values).dtype==np.dtype('float32')
    arm=dict(mean_predicted_gain=display,full_objective=-display,
             unary_objective=-coefficients,product_objective=0.)
    gap=abs(display-coefficients)
    assert 1e-12<gap<=display_roundoff_bound(arm,len(values))
    assert display_roundoff_bound(arm,len(values))<1e-5
