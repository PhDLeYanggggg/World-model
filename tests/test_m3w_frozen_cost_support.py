import numpy as np
import pytest
from src.evaluation import m3w_frozen_cost_support as api


def packet():
    y=np.zeros((4,7,5));y[:,:,2]=10.;y[:,:,3]=10.
    y[0,0,0]=4.;y[1,1,0]=3.;y[2,2,0]=2.;y[3]=np.nan
    p=np.ones_like(y);p[:,:,0]=2;p[:,:,1]=0;p[:,:,2]=10;p[:,:,3]=10;p[:,:,4]=0
    p[0,0,0]=0.;p[0,0,1]=1.
    p[1,1,1]=1.
    pack=dict(y=y,envelope=np.full((4,7),5.),allowed=np.ones((4,7),bool),
        recordings=np.repeat('rec',4),frames=np.zeros(4,int),ids=np.arange(4))
    controls={a:np.array([0,1,2,-1]) for a in api.CONTROLS}
    return pack,p,controls


def test_nested_cost_gates_separate_positive_and_pointwise_loss():
    pack,p,c=packet();r=api.diagnose(pack,p,c)
    assert r['supports']['causal_available']['positive_benefit_upper_mass']==9.
    assert r['supports']['predicted_positive']['positive_benefit_upper_mass']==5.
    assert r['supports']['pointwise_cost_guard']['positive_benefit_upper_mass']==2.
    assert r['supports']['pointwise_cost_guard']['witness']['selected_unknown']==0
    assert r['pointwise_witness_all_original_query_budgets_verified']
    assert not r['new_training'] and r['labels_used_by_oracle']


def test_unknown_and_empty_risk_never_become_safe_labels():
    pack,p,c=packet();pack['y'][:]=np.nan
    with pytest.raises(ValueError):api.diagnose(pack,p,c)
    pack,p,c=packet();pack['allowed'][:]=False
    c={a:np.full(4,-1,int) for a in api.CONTROLS}
    r=api.diagnose(pack,p,c)
    for arm in api.SUPPORTS:
        assert r['supports'][arm]['witness']['easy_selected_risk_upper'] is None
        assert r['supports'][arm]['witness_supported_records']==0


def test_pointwise_guard_is_not_an_upper_bound_on_joint_pooling():
    pack,p,c=packet()
    # A profitable action fails pointwise but fits a pooled budget with a safe anchor.
    p[1,1,1]=.3;p[2,2,2]=100.
    mask=api.candidate_masks(p,pack['allowed'])
    assert not mask['pointwise_cost_guard'][1,1]
    joint=np.array([-1,1,2,-1]);assert api.allocation.feasible(p,pack['allowed'],joint)
    assert api.diagnose(pack,p,c)['pointwise_guard_is_restriction_not_joint_feasible_set']


def test_actual_record_scores_equal_original_controls():
    pack,p,c=packet();r=api.diagnose(pack,p,c)
    for a in api.CONTROLS:
        assert r['controls'][a]==api.compact_score(pack['y'],pack['envelope'],c[a])
        assert r['recordings']['rec']['controls'][a]['actual']==r['controls'][a]


def test_all_original_recordings_remain_when_one_has_no_easy_reference():
    pack,p,c=packet();pack['recordings']=np.array(['a','b','a','b']);pack['y'][1,:,3:]=0
    r=api.diagnose(pack,p,c)
    assert set(r['recordings'])=={'a','b'}
    assert not r['recordings']['b']['causal_available']['oracle']['budget_feasible']
    assert r['supports']['causal_available']['unsupported_record_reasons']['b']=='no_real_allowed_known_easy_reference'


def test_oracle_diagnostic_rejects_missing_controls():
    pack,p,c=packet();c.pop('selected_joint')
    with pytest.raises(ValueError):api.diagnose(pack,p,c)
