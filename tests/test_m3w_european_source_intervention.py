import numpy as np
import pytest

from src.world_model.m3w_european_source_intervention import (
    nested_producers, causal_cost_features, query_subset, controls, paired_cost_labels)


def test_nested_producers_exclude_outer_and_predicted_fold():
    for outer in range(3):
        plan = nested_producers(outer)
        assert set(plan) == set(range(3))-{outer}
        for predicted, trained in plan.items():
            assert trained != outer and trained != predicted
    with pytest.raises(ValueError):
        nested_producers(3)


def test_features_are_past_scaled_and_no_label_api():
    g = np.zeros((2,476),np.float32)
    g[:, :16] = np.tile(np.arange(-7,1),2)
    g[:,16:24] = np.arange(-7,1)/12
    b = np.ones((2,12,2))*2
    p = b+1
    x,s = causal_cost_features(g,b,p)
    assert x.shape == (2,355) and s.shape == (2,)
    x2,s2 = causal_cost_features(g*1,b,p)
    np.testing.assert_array_equal(x,x2)
    assert np.isfinite(x).all() and (s>=1).all()
    with pytest.raises(ValueError):
        causal_cost_features(g,b,np.full_like(p,np.nan))


def test_query_selection_keeps_all_agents_and_is_order_independent():
    sites = np.array(['a','a','a','a','b','b'])
    rec = np.array([0,0,0,1,2,2])
    frames = np.array([10,10,20,30,10,20])
    chosen = query_subset(sites,rec,frames,1,'fixed')
    assert chosen[0] == chosen[1]
    order = np.array([5,2,0,4,1,3])
    reordered = query_subset(sites[order],rec[order],frames[order],1,'fixed')
    np.testing.assert_array_equal(reordered,chosen[order])
    assert chosen[sites=='b'].sum()==1


def test_unknown_labels_remain_unknown_and_positive_costs_exact():
    y = paired_cost_labels(np.array([3.,2.,np.nan]),np.array([1.,4.,np.nan]))
    np.testing.assert_equal(y,np.array([[2.,0.],[0.,2.],[np.nan,np.nan]]))
    with pytest.raises(ValueError):
        paired_cost_labels(np.array([np.nan]),np.array([1.]))


def test_controls_feasible_and_count_matched_without_targets():
    b = np.zeros((2,12,2)); b[1,:,0] = 10
    p = b.copy(); p[:, :,1] += 1
    result = controls(np.array([[2.,.01],[1.,.01]]),np.ones(2,bool),
        np.array([[0.,0.],[10.,0.]]),np.array([2.,2.]),b,p,
        budget=.02,pair_weight=.1,radius_widths=3,threshold_widths=.5,seconds=2)
    assert result['independent']['switch'].all()
    assert result['matched_nonzero']
    for arm in ('independent','scene_uniform','joint','unary_exact','joint_exact'):
        assert result[arm]['predicted_constraints_satisfied']
    assert result['unary_exact']['switch'].sum()==result['joint_exact']['switch'].sum()==2
    no = controls(np.array([[0.,1.],[0.,2.]]),np.ones(2,bool),
        np.array([[0.,0.],[10.,0.]]),np.array([2.,2.]),b,p,
        budget=.02,pair_weight=.1,radius_widths=3,threshold_widths=.5,seconds=2)
    assert not no['independent']['switch'].any() and not no['matched_nonzero']


def test_control_batch_report_is_serializable_and_preserves_unknown_population(monkeypatch):
    import json
    from scripts import run_m3w_european_source_intervention as runner
    monkeypatch.setattr(runner,'beat',lambda *args,**kwargs:None)
    h = np.tile(np.arange(-7,1)[None,:,None],(3,1,2)).astype(float)
    current = np.array([[0.,0.],[10.,0.],[0.,10.]])
    data = dict(history=h+current[:,None],origin=current,width=np.ones(3)*2,
        recordings=np.array([0,0,1]),frames=np.array([10,10,20]),sites=np.array(['a','a','b']))
    a = dict(b=np.ones((3,12,2)),p=np.ones((3,12,2))*1.1)
    reg = dict(predicted_positive_harm_budget=.02,pair_weight=.1,
        edge_radius_bbox_widths=3.,proximity_threshold_bbox_widths=.5,solver_seconds=2.)
    choices,report = runner.decisions(reg,data,a,np.arange(3),np.tile([2.,.001],(3,1)),np.ones(3,bool),1.)
    assert choices['ids'].tolist()==[0,1,2]
    assert len(report)==2
    json.dumps(report,allow_nan=False)


def test_completed_artifact_check_resolves_relative_path_and_detects_mutation(tmp_path,monkeypatch):
    from scripts import run_m3w_european_source_intervention as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    path = tmp_path/'saved.bin'
    path.write_bytes(b'fixed')
    r = {'artifacts':{'checkpoint':{'path':'saved.bin','sha256':runner.digest(path)}}}
    assert runner.artifacts_ok(r)
    path.write_bytes(b'changed')
    assert not runner.artifacts_ok(r)
