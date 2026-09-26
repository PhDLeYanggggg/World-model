import numpy as np
import pytest
from src.evaluation.m3w_cost_factor_attribution import diagnose


def values():
    return [np.array(v,float) for v in ([.2,.8,.4],[2,3,0],[4,1,0],[5,5,0],[1,4,np.nan],
                                       [1,3,np.nan],[.5,.1,0],[2,2,0])]


def test_exact_parts_and_label_counterfactuals():
    p,m,n,env,cv,h,old_e,old_a=values()
    d=diagnose(p,m,n,env,cv,h,2,old_e,old_a,2)['subsets']['all']
    e=np.array([1,0]); err=p[:2]*m[:2]-e*h[:2]
    assert d['rows']==2 and d['unknown_rows']==1
    assert d['easy']['MSE']==pytest.approx(np.mean(err**2))
    assert d['easy']['label_assisted_E_MSE']==pytest.approx(.5)
    assert d['easy']['cross_term']<0
    assert d['strata']['outside_easy_high_severity']['rows']==1
    assert d['easy']['membership_squared']+d['easy']['severity_squared']+d['easy']['cross_term']==pytest.approx(d['easy']['MSE'])
    assert d['all_harm']['membership_squared']+d['all_harm']['severity_squared']+d['all_harm']['cross_term']==pytest.approx(d['all_harm']['MSE'])


def test_exact_zero_cv_is_not_easy_and_zero_envelope_subset():
    z=np.zeros(3); d=diagnose(np.ones(3)*.5,z,z,z,z,z,1,z,z,0)
    assert d['subsets']['all']['strata']['easy']['rows']==0
    assert d['subsets']['all']['easy']['gain_vs_original_percent'] is None
    assert d['subsets']['all']['severity_weighted_Brier'] is None
    assert d['subsets']['disagreement']['status']=='not_estimable'


def test_weights_and_unknowns_do_not_change_other_rows():
    v=values(); d=diagnose(*v[:6],2,*v[6:],2,weights=np.array([1,0,100]))
    assert d['subsets']['all']['rows']==1
    assert d['subsets']['all']['easy']['MSE']==pytest.approx(.36)
    assert d['subsets']['all']['strata']['outside_easy']['mean_probability'] is None


def test_invalid_probabilities_and_unknown_mismatch_fail():
    v=values(); v[0][0]=1.1
    with pytest.raises(ValueError): diagnose(*v[:6],2,*v[6:],2)
    v=values(); v[5][2]=0
    with pytest.raises(ValueError): diagnose(*v[:6],2,*v[6:],2)
