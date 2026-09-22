import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.m3w_cross_objective_review import choices, empirical_gate
from scripts.run_m3w_cross_objective_review import validate_config


def sample():
    a=np.array([[10.,.1],[8.,.2],[6.,.1],[4.,.1],[2.,.1],[1.,.3]])
    b=a.copy();c=a.copy();b[0,1]=2.;c[2,1]=1.
    past=np.zeros((6,8,2));past[:,-1,0]=1
    return a,b,c,past,np.full(6,20.),np.arange(6)


def test_equal_count_does_not_copy_review_decisions():
    a,b,c,past,d,ids=sample();r=choices(a,b,c,past,d,ids)
    np.testing.assert_array_equal(np.flatnonzero(r['nomination']),[0,1,2,3,4])
    np.testing.assert_array_equal(np.flatnonzero(r['reviewed']),[1,3,4])
    np.testing.assert_array_equal(np.flatnonzero(r['matched_nomination']),[0,1,2])
    assert all(np.all(~r[p]|r['nomination']) for p in ('reviewed','matched_nomination'))
    assert r['reviewed'].sum()==r['matched_nomination'].sum()


def test_no_outcome_argument_and_permutation_stable_ties():
    a,b,c,past,d,ids=sample();a[:,0]=10.;a[:,1]=.1
    b[:,1]=[2,.1,2,.1,2,.1];c=b.copy()
    r=choices(a,b,c,past,d,ids);perm=np.array([5,0,4,1,3,2])
    rr=choices(a[perm],b[perm],c[perm],past[perm],d[perm],ids[perm])
    for p in r:np.testing.assert_array_equal(r[p][perm],rr[p])
    assert list(inspect.signature(choices).parameters)==[
        'nomination_score','native_score','fraction_score','past','distance','ids']


@pytest.mark.parametrize('kind',['review_rejects_all','stopped','no_disagreement'])
def test_empty_support_never_forces_switches(kind):
    a,b,c,past,d,ids=sample()
    if kind=='review_rejects_all':b[:,1]=100
    elif kind=='stopped':past[:]=0
    else:d[:]=0
    r=choices(a,b,c,past,d,ids)
    assert not r['reviewed'].any() and not r['matched_nomination'].any()


def test_identity_review_and_invalid_inputs():
    a,b,c,past,d,ids=sample();r=choices(a,a,a,past,d,ids)
    for p in r:np.testing.assert_array_equal(r[p],r['nomination'])
    for bad in (np.full_like(b,np.nan),-b,b[:-1]):
        with pytest.raises(ValueError):choices(a,bad,c,past,d,ids)
    with pytest.raises(ValueError):choices(a,b,c,past,d,np.zeros(6,int))


def test_frozen_configuration_and_no_secondary_winner():
    c=json.loads(Path('configs/m3w_cross_objective_review_v1.json').read_text())
    p=json.loads(Path('configs/m3w_conditional_cost_v1.json').read_text())
    validate_config(c,p)
    for k,v in [('harm_ratio',.2),('reviewers',['native']),('model_selection',True),
                ('risk_calibration',True),('new_training',True),('closed_role_readout',True)]:
        with pytest.raises(ValueError):validate_config(c|{k:v},p)


def test_gate_requires_more_than_reduced_count_or_pooled_easy():
    s=dict(ADE=dict(scene_bootstrap_ci95=[1,3]),seeds={str(seed):dict(
        ADE=dict(equal_scene_gain_percent=2),zero_CV_harmed=0,
        subsets=dict(positive_easy=dict(equal_scene_gain_percent=1,
            by_scene={'a':dict(gain_percent=-1)}))) for seed in (17,29,43)})
    assert all(empirical_gate(s,dict(ci95_pp=[.1,1])).values())
    assert not all(empirical_gate(s,dict(ci95_pp=[-.1,1])).values())
    s['seeds']['17']['subsets']['positive_easy']['by_scene']['a']['gain_percent']=-3
    g=empirical_gate(s,dict(ci95_pp=[.1,1]))
    assert g['aggregate_easy'] and not g['each_scene_seed_easy']
