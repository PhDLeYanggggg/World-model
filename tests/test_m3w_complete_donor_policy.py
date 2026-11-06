import numpy as np
import pytest

from src.evaluation import m3w_complete_donor_policy as m


def test_no_proposal_is_never_created_by_a_zero_padded_score():
    proposal = np.array([-1,1,3])
    risk = np.array([[0.,0.],[-1.,-1.],[-1.,.1]])
    np.testing.assert_array_equal(m.veto(proposal,risk),[-1,1,-1])


def test_identity_fallback_with_no_support_remains_NA():
    floor = np.zeros((2,12,2))
    target = np.ones_like(floor)
    p = m.original.policy(floor,floor,floor,target,np.ones((2,12),bool),np.zeros(2,int),easy_cut=2,hard_cut=3)
    result = m.safety([p])
    assert result['undefined_all_views']==1 and result['undefined_easy_views']==1
    assert not result['original_record_contract_pass']
    assert not m.safety([])['original_record_contract_pass']


def test_unknown_selected_path_enters_the_original_upper_bound():
    floor = np.zeros((2,12,2))
    prediction = floor.copy()
    prediction[1,:,0] = 5
    prediction[0,:,0] = 1
    target = np.ones_like(floor)
    valid = np.ones((2,12),bool)
    valid[1] = False
    p = m.original.policy(prediction,floor,floor,target,valid,np.zeros(2,int),easy_cut=2,hard_cut=3)
    r = p['recordings']['0']
    assert r['unknown_selected']==1
    assert r['completion_all_risk_upper']>r['known_all_risk']
    assert not r['supported']


def test_future_outcomes_not_accepted_by_veto():
    with pytest.raises(TypeError):
        m.veto(np.array([1]),np.zeros((1,2)),target=np.ones((1,12,2)))


def test_same_count_empty_utility_is_not_safe():
    from src.evaluation.m3w_matched_intervention_utility import matched_utility
    floor = np.zeros((2,12,2))
    c = np.full(2,-1,int)
    u = matched_utility(floor,floor,floor,np.ones_like(floor),np.ones((2,12),bool),c,c,[np.arange(2)])
    assert u['status']=='not_estimable_empty_matched_control'
    assert not u['safety_gate_usage']
