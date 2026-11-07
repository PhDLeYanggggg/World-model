import numpy as np

from src.world_model import m3w_shortlist_cost as model


def test_risk_changes_choice_in_fixed_shortlist():
    sample = (np.array([0,1]),np.array([0,1]),np.array([2,3]),np.eye(2))
    co = np.array([[.6,.5,.5,.2,.9],[.4,-.2,-.1,-.05,.7]])
    a,b = model.winners(sample,co)
    assert a==(.9,0,2) and b==(.7,1,3)


def test_targets_distinguish_positive_supported_from_harm():
    z = np.array([[[1.,1.,0.,0.,-.2,.2]],[[1.,1.,.1,.1,.1,-.1]]])
    q = (np.arange(2),np.arange(2),np.zeros(2,int),np.eye(2))
    a = model.targets(z,np.ones(2),[q])[0]
    np.testing.assert_allclose(a,[[.2,-.02,-.02,-.2,1.],[-.1,.08,.08,.1,0.]])
