import numpy as np
from src.evaluation import m3w_easy_constrained_policy as api


def test_empty_matched_risk_remains_undefined_not_a_safety_pass():
    y=np.tile([.1,0.,1.,1.,0.],(4,1));env=np.ones(4)
    p={a:y.copy() for a in api.ARMS};p['origin'][:,0]=0
    args=(np.zeros((4,2)),env,y,None,None,np.array([0,0,1,1]),np.arange(4),{})
    result,action=api.evaluate(args,p,np.ones(4,bool))
    m=result['matched']['origin'];assert m['empty'] and m['empty_risk_status']=='not_estimable'
    assert m['left']['easy_selected_risk_upper'] is None and m['right']['easy_selected_risk_upper'] is None
    assert result['contrasts']['origin_matched']==0 and result['contrasts']['origin_full']>0
    assert not result['matched_risk_required']
    assert not api.budget_status(m['right'])['supported']
    y[:]=np.nan;args=(args[0],env,y,*args[3:])
    _,changed=api.evaluate(args,p,np.ones(4,bool))
    for a in api.ARMS:np.testing.assert_array_equal(action[a],changed[a])


def test_positive_utility_cannot_rescue_excess_full_policy_risk():
    p=dict(all_selected_risk_upper=.03,easy_selected_risk_upper=.01,easy_degradation_upper=-.02)
    assert not api.budget_status(p)['supported']
    p['all_selected_risk_upper']=.01;assert api.budget_status(p)['supported']
    p['easy_selected_risk_upper']=None;assert not api.budget_status(p)['supported']
