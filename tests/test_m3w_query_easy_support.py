import itertools
import numpy as np
import pytest
from src.evaluation import m3w_query_easy_support as api


def costs(n):
    p=np.zeros((n,7,5));p[:,:,2:4]=10.
    return p,np.ones((n,7),bool)


def test_singleton_and_nonnegative_sign_obstructions_are_exact():
    p,ok=costs(1);p[:,:,1]=.3
    r,c=api.query_witness(p,ok,np.array([True]),seconds=2.)
    assert r['status']=='infeasible' and r['reason']=='singleton_exact_no_anchor' and (c==-1).all()
    p,ok=costs(3);p[:,:,1]=.3;p[1:,:,1]=.2
    r,_=api.query_witness(p,ok,np.array([True,False,False]),seconds=2.)
    assert r['status']=='infeasible' and r['reason']=='nonnegative_cost_sign_certificate'


def test_pooling_can_recover_an_anchor_that_fails_pointwise():
    p,ok=costs(2);p[0,:,1]=.3
    assert not api.support.pointwise_actions(p,ok)[0].any()
    r,c=api.query_witness(p,ok,np.array([True,False]),seconds=2.)
    assert r['status']=='feasible' and r['reason']=='exact_budget_postchecked_milp'
    assert api.support.feasible(p,ok,c) and c[0]>=0 and c[1]>=0


def test_small_queries_match_exhaustive_eight_action_search():
    rng=np.random.default_rng(271828)
    for _ in range(30):
        p,ok=costs(3);p[:,:,1]=rng.uniform(0,.45,(3,7));p[:,:,4]=p[:,:,1]*rng.uniform(0,1,(3,7))
        p[:,:,3]=rng.uniform(.5,10,(3,7));ok[:]=rng.random((3,7))>.4;easy=np.array([True,False,True])
        brute=any(api.support.feasible(p,ok,np.array(c)) and easy[np.flatnonzero(np.array(c)>=0)].any()
            for c in itertools.product(range(-1,7),repeat=3))
        r,c=api.query_witness(p,ok,easy,seconds=2.)
        assert r['status']!='inconclusive'
        assert (r['status']=='feasible')==brute


def test_binary_anchor_support_handles_tiny_native_reference_without_epsilon():
    p,ok=costs(2);p*=1e-120;p[0,:,1]=.3e-120
    r,c=api.query_witness(p,ok,np.array([True,False]),seconds=2.)
    assert r['status']=='feasible' and api.support.feasible(p,ok,c)


def test_record_retains_every_query_and_does_not_claim_actual_safe_gain():
    p,ok=costs(2);p[:,:,1]=.3;y=np.zeros((2,7,5));y[:,:,2:4]=1.
    packet=dict(y=y,allowed=ok,recordings=np.repeat('r',2),frames=np.arange(2),ids=np.arange(2))
    r=api.recording_diagnosis(packet,p,seconds=2.)
    assert r['queries']==2 and r['complete_easy_support_obstruction'] and r['obstruction_algebraic_or_singleton_only']
    assert not r['positive_actual_safe_gain_proven']


def test_known_easy_mask_and_runtime_are_validated():
    p,ok=costs(1)
    with pytest.raises(ValueError):api.query_witness(p,ok,np.array([1]),seconds=2.)
    with pytest.raises(ValueError):api.query_witness(p,ok,np.array([True]),seconds=0.)
