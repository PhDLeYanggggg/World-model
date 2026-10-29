import numpy as np
import pytest

from src.evaluation import m3w_policy_conditional_readout as m


def example():
    y=np.array([[.3,0.,1.,1.,0.],[0.,.05,1.,1.,.05],[np.nan]*5])
    p=np.tile([.2,.001,1.,.5,.001],(3,1));env=np.ones(3)
    masks=dict(raw=np.array([1,1,1],bool),global_=np.array([1,1,0],bool),
        support_only=np.array([1,0,1],bool),conditional=np.array([1,0,0],bool))
    masks['global']=masks.pop('global_')
    return y,env,np.array(['a','a','b']),np.arange(3),p,masks


def test_aggregate_verifies_and_unknowns_count_against_support():
    r=m.evaluate(*example());m.validate(r)
    assert r['policies']['raw']['selected_unknown']==1
    assert r['policies']['conditional']['finite_completion_supported']
    assert not r['policies']['raw']['finite_completion_supported']
    assert r['pairs']['raw_full']['unknown_exchanged_envelope_mass']==1


def test_empty_policy_is_not_safe_success():
    args=list(example());args[-1]['conditional'][:]=False
    r=m.evaluate(*args);m.validate(r)
    assert r['policies']['conditional']['easy_selected_risk_upper'] is None
    assert not r['policies']['conditional']['finite_completion_supported']


def test_missing_view_not_silently_dropped():
    with pytest.raises(ValueError,match='Complete72'):m.summarize([],[])
