import itertools

import numpy as np
import pytest

from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_easy_allocation import allocate, selected_budget
from src.world_model.m3w_interaction_controls import decompose_pair_objective


def problem(gain, support=None, pairs=None):
    gain=np.asarray(gain,float);n=len(gain)
    edges=np.array([[0,1]],dtype=np.int64) if pairs is not None else np.empty((0,2),np.int64)
    return InterventionProblem(gain,np.zeros(n),np.ones(n,bool) if support is None else support,
        edges,np.array([pairs],float) if pairs is not None else np.zeros((0,2,2)),1.,100.,n)


def test_pool_denominator_not_silently_selected_denominator():
    p=problem([3,1,0],np.array([True,True,False]))
    q=np.array([1.,1.,0.]);r=np.array([1.,1.,100.])
    bits,info=allocate(p,q,r,np.ones(3,bool),cost_scale=1,
        pointwise=np.zeros(3,bool),strict=np.zeros(3,bool))
    assert not bits['aggregate_selected'].any()
    assert bits['aggregate_population'].tolist()==[True,True,False]
    assert not bits['scene_uniform'].any()
    assert info['population_budget']==pytest.approx(2.04)


def test_signed_selected_budget_recovers_offsetting_allowance():
    gain=np.array([3.,2.,1.]);q=np.array([.4,0.,1.]);r=np.array([1.,30.,1.])
    b,ok,_=selected_budget(gain,q,r,np.ones(3,bool),.02,5.)
    assert ok and b.tolist()==[True,True,False]
    assert q@b<=.02*(r@b)


@pytest.mark.parametrize('seed',range(8))
def test_selected_milp_matches_complete_enumeration(seed):
    rng=np.random.default_rng(seed);n=6
    gain=rng.random(n)+.01;q=rng.random(n);r=rng.random(n)*40
    support=rng.random(n)>.2
    b,ok,_=selected_budget(gain,q,r,support,.02,5.)
    possible=[np.array(x,bool) for x in itertools.product((False,True),repeat=n)]
    expected=max(float(gain@x) for x in possible if not (x&~support).any() and (q-.02*r)@x<=1e-10)
    assert ok and gain@b==pytest.approx(expected,abs=1e-8)


def test_population_and_geometry_bruteforce_same_budget_count():
    p=problem([3,2,1],pairs=[[0,.1],[.1,.9]])
    q=np.array([.2,.2,.2]);r=np.array([5.,5.,5.])
    chosen,report=allocate(p,q,r,np.ones(3,bool),cost_scale=1,
        pointwise=np.zeros(3,bool),strict=np.zeros(3,bool))
    assert chosen['aggregate_population'].tolist()==[True,False,False]
    assert report['matched']
    for name in ('aggregate_population','aggregate_unary','aggregate_joint'):
        assert chosen[name].sum()==1 and q@chosen[name]<=.02*r.sum()


def test_zero_budget_fallback_and_free_positive_candidate():
    p=problem([2.,1.]);q=np.array([0.,1.]);r=np.zeros(2)
    b,_=allocate(p,q,r,np.ones(2,bool),cost_scale=1,
        pointwise=np.zeros(2,bool),strict=np.zeros(2,bool))
    assert b['aggregate_population'].tolist()==[True,False]


def test_non_target_context_cannot_supply_risk_allowance():
    p=problem([1.,0.],np.array([True,False]));target=np.array([True,False])
    with pytest.raises(ValueError):
        allocate(p,np.array([1.,0.]),np.array([1.,100.]),target,cost_scale=1,
                 pointwise=np.zeros(2,bool),strict=np.zeros(2,bool))


@pytest.mark.parametrize('seed',range(4))
def test_population_and_matched_pair_objectives_exhaustive(seed):
    rng=np.random.default_rng(seed+50);n=5
    p=problem(rng.random(n)+.01,pairs=[[0,.8],[.2,.05]])
    q=rng.random(n)+.1;r=np.full(n,10.)
    choices,info=allocate(p,q,r,np.ones(n,bool),cost_scale=1,
        pointwise=np.zeros(n,bool),strict=np.zeros(n,bool))
    possible=[np.array(x,bool) for x in itertools.product((False,True),repeat=n) if q@np.array(x)<=.02*r.sum()]
    best=max(p.expected_gain@x for x in possible)
    assert p.expected_gain@choices['aggregate_population']==pytest.approx(best,abs=1e-8)
    k=int(choices['aggregate_population'].sum());possible=[x for x in possible if x.sum()==k]
    parts=decompose_pair_objective(p)
    for name,full in [('aggregate_unary',False),('aggregate_joint',True)]:
        def objective(x):
            return parts['unary_objective']@x+(parts['product_objective']@(x[p.edges[:,0]]&x[p.edges[:,1]]) if full else 0)
        assert objective(choices[name])==pytest.approx(min(objective(x) for x in possible),abs=1e-8)
    assert info['matched']


def test_fixed_tolerance_cannot_be_relaxed():
    with pytest.raises(ValueError):
        allocate(problem([1.,1.]),np.ones(2),np.ones(2),np.ones(2,bool),cost_scale=1,
                 pointwise=np.zeros(2,bool),strict=np.zeros(2,bool),rho=.03)
