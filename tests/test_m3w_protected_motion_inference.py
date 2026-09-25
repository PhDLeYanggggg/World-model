import numpy as np

from src.world_model.m3w_european_protected_motion import candidate_decisions


def inference_fixture():
    origin=np.array([[0.,0.],[1.,0.],[50.,0.],[51.,0.]])
    history=origin[:,None]+np.column_stack((np.arange(-7,1),np.zeros(8)))[None]
    b=np.broadcast_to(np.column_stack((np.arange(1,13),np.zeros(12))), (4,12,2)).copy()
    data=dict(history=history,origin=origin,width=np.ones(4),
        recordings=np.array([1,1,2,2]),frames=np.array([10,10,20,20]),
        sites=np.array(['a','a','b','b']),valid=np.ones((4,12),bool),target_eval=np.zeros((4,12,2)))
    a=dict(b=b,p=b*.9)
    reg=dict(predicted_risk_budget=.02,pair_weight=.1,edge_radius_bbox_widths=3.,
        proximity_threshold_bbox_widths=.5,solver_seconds=2.)
    return reg,data,a


def test_future_poison_and_missing_labels_do_not_change_decisions():
    reg,data,a=inference_fixture()
    held=np.arange(4);mask=np.ones(4,bool)
    utility=np.array([1.,2.,3.,4.]);moments=np.array([[1.,0.],[1.,.01],[1.,0.],[1.,.01]])
    before,queries_before=candidate_decisions(reg,data,a,held,mask,utility,moments,True)
    poisoned=dict(data,valid=np.zeros((4,12),bool),target_eval=np.full((4,12,2),np.nan),
        future_endpoint=np.full((4,2),1e100),oracle_choice=np.zeros(4,int))
    after,queries_after=candidate_decisions(reg,poisoned,a,held,mask,utility,moments,True)
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    assert queries_before==queries_after
    assert len(after['ids'])==4 and sum(q['agents'] for q in queries_after)==4


def test_source_support_guard_preserves_population_but_abstains():
    reg,data,a=inference_fixture()
    choices,queries=candidate_decisions(reg,data,a,np.arange(4),np.ones(4,bool),
        np.ones(4),np.column_stack((np.ones(4),np.zeros(4))),False)
    assert len(choices['ids'])==len(choices['pointwise_ids'])==4
    for key in ('pointwise','independent','scene_uniform','joint','unary_exact','joint_exact'):
        assert not choices[key].any()
    assert all(q['matched'] and not q['matched_nonzero'] and q['reference_count']==0 for q in queries)
