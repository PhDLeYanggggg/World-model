from itertools import combinations
import numpy as np
import pytest
from src.world_model.m3w_query_utility import allocate, grouped


def test_budget_pooling_recovers_more_utility_at_equal_count():
    u=np.array([1.,2.,5.]); q=np.array([[-2.,-2.],[-2.,-2.],[1.,1.]])
    a=np.array([True,True,False])
    out,info=allocate(u,q,np.ones(3,bool),a,np.arange(3))
    assert out['joint_utility'].tolist()==[False,True,True]
    assert out['joint_utility'].sum()==a.sum()
    assert info['predicted_feasible'] and info['expected_utility_gain']==4


def test_integer_solver_matches_exhaustive_small_problem():
    rng=np.random.default_rng(41)
    for _ in range(12):
        q=rng.uniform(-1,2,(6,2)); q[:2]=-abs(q[:2])
        u=rng.uniform(.2,5,6); a=np.arange(6)<2
        out,info=allocate(u,q,np.ones(6,bool),a,np.arange(6))
        best=max(sum(u[list(ix)]) for ix in combinations(range(6),2) if np.all(q[list(ix)].sum(0)<=1e-10))
        assert not info['fallback']
        assert u@out['joint_utility']==pytest.approx(best)


def test_solver_failure_preserves_frozen_anchor(monkeypatch):
    from types import SimpleNamespace
    import src.world_model.m3w_query_utility as api
    monkeypatch.setattr(api,'milp',lambda **kw:SimpleNamespace(success=False,x=None,fun=None))
    a=np.array([True,False]); out,info=allocate(np.array([1.,9.]),np.array([[-1.,-1.],[2.,2.]]),
        np.ones(2,bool),a,np.arange(2))
    np.testing.assert_array_equal(out['joint_utility'],a)
    assert info['fallback'] and not info['optimal']


def test_no_borrowing_across_current_frames():
    u=np.array([1.,2.,10.,20.]); q=np.array([[-1.,-1.],[-1.,-1.],[1.,1.],[1.,1.]])
    a=np.array([True,True,False,False])
    out,_=grouped(u,q,np.ones(4,bool),a,np.array(['r']*4),np.array([1,1,2,2]),np.arange(4))
    np.testing.assert_array_equal(out['joint_utility'],a)


def test_ineligible_agent_blocks_uniform_but_not_valid_subset():
    out,_=allocate(np.array([1.,0.]),-np.ones((2,2)),np.array([True,False]),np.array([True,False]),np.arange(2))
    assert not out['query_uniform'].any()
    assert out['joint_utility'].tolist()==[True,False]


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError,match='Causal'):
        allocate(np.ones(2),-np.ones((2,2)),np.ones(2,bool),np.ones(2,bool),np.zeros(2,int))
