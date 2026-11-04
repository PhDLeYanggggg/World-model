"""Exhaustively check the diagnostic criterion against the unchanged evaluator."""
import itertools
import numpy as np
from src.evaluation.m3w_record_risk_feasibility import witness
from src.evaluation.m3w_selection_forecast_readout import policy


def test_criterion_equals_exhaustive_binary_policy_search():
    rng=np.random.default_rng(20261104)
    for _ in range(30):
        n=5
        reference=rng.choice([0.,1.,10.,100.],n)
        candidate=np.maximum(0.,reference+rng.choice([-3.,0.,.01,.03,20.],n))
        cv=rng.choice([0.,1.,3.],n)
        known=rng.random(n) > .25
        allowed=reference != candidate
        rec=np.full(n,'a')
        costs=[np.where(known,v,np.nan)for v in (reference,candidate,cv)]
        choices,result=witness(*costs,allowed,rec,easy_cut=2.)
        floor=np.zeros((n,12,2)); floor[...,0]=reference[:,None]
        proposal=np.zeros_like(floor); proposal[...,0]=candidate[:,None]
        cv_path=np.zeros_like(floor); cv_path[...,0]=cv[:,None]
        valid=np.broadcast_to(known[:,None],(n,12)).copy()
        target=np.where(valid[...,None],np.zeros_like(floor),np.nan)
        def passed(take):
            prediction=np.where(take[:,None,None],proposal,floor)
            return policy(prediction,floor,cv_path,target,valid,rec,easy_cut=2.,hard_cut=2.)['recordings']['a']['supported']
        possible=any(passed(np.array(bits,bool)&allowed)for bits in itertools.product((False,True),repeat=n))
        assert possible == result['a']['binary_fixed_proposal_finite_risk_feasible']
        assert passed(choices) == possible
