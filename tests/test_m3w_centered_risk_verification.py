import numpy as np
import pytest
from scripts.verify_m3w_centered_risk_policy import validate_actions


def fixture():
    a={'ids':np.arange(3),'eligible':np.ones(3,bool)}
    for key,value in [('raw_independent',[1,1,0]),('centered_independent',[1,1,0]),
                      ('centered_joint',[1,1,0]),('matched_raw_joint',[0,1,1]),
                      ('centered_utility_topk',[0,1,1]),('centered_query_uniform',[0,0,0])]:
        a['a_'+key]=np.array(value,bool)
    return a,np.array([[-2.,-2.],[-2.,-2.],[1.,1.]]),[.75,.75],np.array([1.,2.,9.]),['s']*3,['r']*3,[0]*3,'a'


def test_independent_validator_accepts_count_matched_change():
    assert validate_actions(*fixture())==(1,1)


def test_validator_rejects_extra_action():
    x=fixture();x[0]['a_matched_raw_joint'][0]=True
    with pytest.raises(AssertionError):validate_actions(*x)


def test_validator_rejects_centered_risk_violation_at_same_count():
    x=fixture();x[0]['a_centered_joint']=x[0]['a_matched_raw_joint'].copy()
    with pytest.raises(AssertionError):validate_actions(*x)


def test_validator_rejects_borrowing_across_frames():
    x=list(fixture());x[6]=[0,0,1]
    with pytest.raises(AssertionError):validate_actions(*x)
