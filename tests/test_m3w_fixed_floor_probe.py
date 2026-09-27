import numpy as np
import pytest
from src.world_model.m3w_fixed_floor_probe import (
    assert_roles, targets, fit_ridge, predict, decisions, accounting)


def test_full_producer_chain_excluded():
    assert_roles(['p0','p1','p2','p3'], ['c0','c1','c2','c3'], ['a','b'], ['d','e'])
    with pytest.raises(ValueError):
        assert_roles(['a','p1','p2','p3'], ['c0','c1','c2','c3'], ['a','b'], ['d','e'])


def test_unknown_stays_unknown_easy_event_unchanged():
    y = targets(np.array([0.,1.,5.,np.nan]), np.array([0.,.5,4.,np.nan]),
                np.array([1.,.2,6.,np.nan]), 2.)
    np.testing.assert_array_equal(y[:3], [[0,1,0,0,0],[.3,0,.5,.5,0],[0,2,4,0,0]])
    assert np.isnan(y[-1]).all()
    with pytest.raises(ValueError): targets(np.array([1.]),np.array([1.]),np.array([np.nan]),2.)


def test_fit_is_source_balanced_and_unknown_excluded():
    x = np.arange(12., dtype=float).reshape(6,2)
    cv = np.array([1.,2,3,4,5,np.nan]); sites = np.array(['a','a','a','a','b','b'])
    y = np.tile(np.arange(5.),(6,1)); y[-1] = np.nan
    model = fit_ridge(x,y,cv,sites,ridge_lambda=.01,clip=10.)
    assert model['training_sites'] == ['a','b'] and model['known_rows'] == 5
    assert np.isclose(model['scale'],3.75)
    assert model['site_weight'] == {'a':.5,'b':.5}
    yy = y.copy(); yy[-1] = 999.
    # A label mask disagreement cannot turn unknown rows into training samples.
    with pytest.raises(ValueError): fit_ridge(x,yy,cv,sites,ridge_lambda=.01,clip=10.)
    xx=x.copy(); xx[-1]=1e8
    other = fit_ridge(xx,y,cv,sites,ridge_lambda=.01,clip=10.)
    np.testing.assert_array_equal(model['coef'],other['coef'])


def test_ridge_closed_form_and_envelope():
    rng=np.random.default_rng(9); x=rng.normal(size=(80,4))
    y=np.column_stack([2+x[:,0]*.1,2-x[:,0]*.1,np.ones(80)*5,np.ones(80),np.ones(80)*.1])
    cv=np.ones(80)*4; sites=np.array(['a']*40+['b']*40)
    model=fit_ridge(x,y,cv,sites,ridge_lambda=.01,clip=10.)
    z=np.clip((x-model['mean'])/model['std'],-10,10)
    manual=np.linalg.solve(z.T@z/80+.01*np.eye(4),z.T@(y/4-model['constant'])/80)
    np.testing.assert_allclose(model['coef'],manual,atol=1e-10)
    p,d=predict(model,x,np.ones(80)*.5)
    assert (p[:,:2].sum(1)<=.5+1e-12).all() and (p[:,4]<=.5).all()
    assert np.isfinite(d).all()


def test_safety_requires_movement_support_and_both_risks():
    p=np.array([[2,.01,1,1,.01],[2,.01,1,1,.03],[2,.03,1,1,.01],[2,.01,1,1,.01]])
    positive,safe=decisions(p,np.array([True,True,True,False]),np.array([True,True,True,True]))
    assert positive.tolist()==[True,True,True,False]
    assert safe.tolist()==[True,False,False,False]
    assert not decisions(p,np.ones(4,bool),np.zeros(4,bool))[1].any()


def test_accounting_rebase_removes_only_default_difference():
    cv=np.array([2.,4.,1.,np.nan]); floor=np.array([1.,3.,2.,np.nan]); neural=np.array([.5,8.,3.,np.nan])
    take=np.array([True,False,False,True]); out=accounting(cv,floor,neural,take)
    assert out['captured_benefit']==.5 and out['selected_harm']==0
    assert out['fallback_regression']==1 and out['fallback_relief']==1
    assert out['missed_benefit']==0 and out['unknown_rows']==1
    assert np.isclose(out['rebased_gain_sum'],out['original_gain_sum'])
    assert out['oracle_benefit']==.5
