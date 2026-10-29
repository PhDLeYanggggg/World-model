import copy
import numpy as np
import pytest
from src.evaluation import m3w_cost_bank_factorization as api
from src.world_model.m3w_multiaction_cost import select_actions


def sample():
    mixed=np.tile([1.,.01,10.,10.,.01],(4,7,1))
    mixed[:,0,0]=5.;mixed[:,6,0]=2.
    mixed[1,6,3]=.1;mixed[3,6,1]=1.;mixed[3,6,3]=.1;mixed[3,6,4]=.5
    neural=mixed.copy();neural[:,6]=[9.,.01,10.,10.,.01]
    allowed=np.ones((4,7),bool);allowed[2,6]=False
    return dict(physical_only=mixed.copy(),physical_neural=mixed,neural_only=neural),allowed


def test_neutral_reconstruction_is_exact_and_inactive_neural_physical_values_do_not_leak():
    p,allowed=sample();original=copy.deepcopy(p)
    neutral=api.factorized(p['physical_neural'],p['physical_neural'])
    np.testing.assert_array_equal(neutral,p['physical_neural'])
    left=select_actions(neutral,allowed,arm='physical_neural')
    right=select_actions(p['physical_neural'],allowed,arm='physical_neural')
    for a,b in zip(left,right):np.testing.assert_array_equal(a,b)
    first=api.factorized(p['physical_only'],p['neural_only'])
    p['physical_only'][:,6]+=1000
    np.testing.assert_array_equal(first,api.factorized(p['physical_only'],p['neural_only']))
    for key in ('physical_neural','neural_only'):np.testing.assert_array_equal(p[key],original[key])


def test_competition_is_separated_from_training_bank_gate_suppression():
    p,allowed=sample();choice,rank,predicted,diagnosis=api.policies(p,allowed)
    assert choice['physical_neural'].tolist()==[0,0,0,0]
    assert choice['mixed_neural_restricted'].tolist()==[6,-1,-1,-1]
    assert choice['competition_lost'].tolist()==[6,-1,-1,-1]
    assert choice['suppressed_neural_expert'].tolist()==[-1,6,-1,6]
    assert choice['factorized'].tolist()==[6,6,0,6]
    assert diagnosis['neural_expert_pass_mixed_gate_reject']==2
    assert diagnosis['mixed_gate_pass_physical_competition_lost']==1
    assert diagnosis['mixed']['eligible']==3 and diagnosis['mixed']['unavailable']==1
    assert diagnosis['mixed']['overlapping_failures']['all_harm_budget']==1
    assert diagnosis['mixed']['overlapping_failures']['easy_harm_budget']==2
    assert sum(diagnosis['mixed']['exclusive_failures'].values())==3
    assert sum(diagnosis['mixed']['failure_bit_patterns'].values())==3
    assert set(predicted)=={'physical_only','physical_neural','neural_only','factorized'}


def test_no_future_label_or_support_argument_at_inference():
    p,allowed=sample()
    with pytest.raises(TypeError):api.policies(p,allowed,future_endpoint=np.zeros((4,2)))
    with pytest.raises(TypeError):api.policies(p,allowed,known_future_mask=np.ones(4,bool))
    with pytest.raises(ValueError):api.factorized(np.full((4,7,5),np.nan),p['neural_only'])
    with pytest.raises(ValueError):api.policies(p,allowed.astype(float))


def test_empty_eligible_set_has_zero_causal_counts_not_a_safety_pass():
    p,allowed=sample();allowed[:]=False
    choice,rank,_,diagnosis=api.policies(p,allowed)
    assert all(np.all(c==-1) for c in choice.values())
    assert diagnosis['mixed']['eligible']==0 and diagnosis['mixed']['unavailable']==4
    assert sum(diagnosis['mixed']['exclusive_failures'].values())==0
    assert 'safety_pass' not in diagnosis


def test_partial_source_roster_cannot_be_summarized_as_complete():
    from scripts.diagnose_m3w_cost_bank_factorization import summarize
    with pytest.raises(ValueError,match='All72fitting'):
        summarize([],dict(seeds=[17,29,43]))
