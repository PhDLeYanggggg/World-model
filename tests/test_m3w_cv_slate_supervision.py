import numpy as np
import pytest

from scripts.audit_m3w_cv_fixed_slate_capacity import recording
from scripts.audit_m3w_cv_joint_feasibility import verify
from src.world_model.m3w_cv_slate_supervision import contributions, menu, preference, verdict


def paths(x):
    out = np.zeros((len(x),12,2))
    out[:,:,0] = np.asarray(x)[:,None]
    return out


def test_replacement_verdict_matches_original_risk_and_dual_gain():
    cv,ols,y = paths([1,1,1,1]),paths([.9,.9,.9,.9]),paths([0,0,0,0])
    good,unsafe = paths([.5,1,.5,1]),paths([1.5,1,.5,1])
    v = np.ones((4,12),bool)
    q = np.repeat([0,1],2)
    w = np.full(4,.5)
    bank = np.stack((good,unsafe),axis=1)
    total,ref,offset,cap = contributions(cv,ols,good,y,v,w,2.)
    old,*_ = contributions(cv[:2],ols[:2],good[:2],y[:2],v[:2],w[:2],2.)
    new,*_ = contributions(cv[:2],ols[:2],unsafe[:2],y[:2],v[:2],w[:2],2.)
    for candidate,expected in ((total,True),(total-old+new,False)):
        arms = [[0,0 if expected else 1],[2,0]]
        outcome = verify(arms,cv,ols,bank,y,v,q,2.,5.,1)
        gain,ok = verdict(candidate,ref,offset,cap)
        assert ok is expected is outcome['verified']
        assert gain == pytest.approx(min(outcome['gain_pp']))


def test_fixed_menu_capacity_keeps_query_counts_and_cannot_certify_missing_support():
    cv,ols,y = paths([1,1,1,1]),paths([.9,.9,.9,.9]),paths([0,0,0,0])
    bank = np.stack((paths([.75,.75,.75,.75]), paths([1.5,1.5,1.5,1.5])),axis=1)
    anchor = np.array([1,-1,1,-1],dtype=np.int8)
    h = dict(target=y,valid=np.ones((4,12),bool),query=np.repeat([0,1],2))
    assert recording(cv,ols,bank,h,anchor,(2.,5.))=='verified'
    assert recording(cv,ols,bank,h,anchor,(.5,5.))=='undefined_support'
    for a in menu(cv,ols,bank,2):
        assert (a>=0).sum()==2
    with pytest.raises(ValueError,match='fixed count'):
        menu(cv,ols,cv[:,None],1)


def test_preferences_distinguish_gain_from_feasibility():
    assert preference((5,False),(2,True),False)==1
    assert preference((5,False),(2,True),True)==-1
    assert preference((5,False),(5,False),True)==0
