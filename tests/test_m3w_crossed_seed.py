import inspect
import numpy as np
import pytest
from src.world_model import m3w_crossed_seed as api


def test_all_seeds_retained_and_screen_only_from_source():
    good=np.tile([1.,0.,2.,2.,0.],(4,1));bad=good.copy();bad[:,1]=2
    p={17:good,29:bad,43:good}
    out=api.actions(p,np.ones(4,bool),np.ones(4,bool),np.array(['r']*4),np.array([1,1,2,2]),np.arange(4),
                    {17:False,29:True,43:True})
    assert out['seed17'].all() and not out['screen17'].any()
    assert not out['seed29'].any() and out['seed43'].all()
    assert not out['ref43_matched29'].any() and out['ref43_matched17'].all()
    with pytest.raises(ValueError,match='All registered'):
        api.actions({17:good},None,None,None,None,None,{17:True})
    assert 'target' not in inspect.signature(api.actions).parameters


def test_accounting_decomposes_harm_and_reference_error():
    y=np.array([[0.,.2,1.,1.,.2],[1.,0.,5.,0.,0.],[np.nan]*5])
    p=np.array([[1.,.01,2.,2.,.01],[1.,.01,5.,3.,.01],[2.,.01,6.,1.,.01]])
    r=api.component_accounting(y,p,np.ones(3,bool))
    assert r['known_selected']==2 and r['unknown_selected']==1
    assert r['actual_easy_risk']==pytest.approx(.2)
    assert r['predicted_easy_risk']==pytest.approx(.02/5)
    assert r['easy_harm_error_contribution']==pytest.approx(.18)
    assert r['easy_reference_error_contribution']==pytest.approx(.08)
    assert r['actual_easy_budget_excess']-r['predicted_easy_budget_excess']==pytest.approx(.26)
    assert not r['used_for_inference']


def test_empty_or_unknown_is_not_a_safety_pass():
    y=np.array([[np.nan]*5]);p=np.ones((1,5));a=np.ones(1,bool)
    r=api.component_accounting(y,p,a)
    assert r['actual_easy_risk'] is None
    assert api.risk_status(r['actual_easy_risk'])=='undefined_not_pass'
    assert api.risk_status(.021)=='violating'
    assert api.risk_status(.02)=='defined_within_budget'
    with pytest.raises(ValueError):api.risk_status(float('nan'))


def test_component_sums_keep_zero_reference_harm():
    r=api.component_accounting(np.array([[0.,2.,0.,0.,2.]]),np.zeros((1,5)),np.ones(1,bool))
    assert r['actual_easy_risk'] is None and r['actual_easy_budget_excess']==2
    with pytest.raises(ValueError):
        api.component_accounting(np.array([[0.,np.nan,1.,0.,0.]]),np.zeros((1,5)),np.ones(1,bool))
