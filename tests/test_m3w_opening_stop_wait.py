import numpy as np

from src.world_model import m3w_opening_stop_wait as model


def test_first_decision_ignores_next_features_and_uses_continuation():
    a = (0.,0,0,np.array([1.,0.]))
    b = (0.,1,0,np.array([1.,1.]))
    c = (0.,2,0,np.array([1.,-1.]))
    stop,wait = np.array([.3,.5]),np.array([.5,0.])
    s,w = model.plan([a,b],stop,wait)
    assert s[0][:3]==(.3,0,0) and w[0][:3]==(.5,1,0)
    assert model.plan([a],stop,wait)==(s,[])
    assert model.plan([a,c],stop,wait)[1][0][1]==2


def test_continuation_uses_realized_stop_return_not_hindsight_maximum():
    x = np.ones((2,1))
    s,w = model.fit({'a':[(x,np.array([1.,-1.]))],
                     'b':[(x,np.array([1.,1.]))]},1)
    assert s[0]==.5
    assert w[0]==-.25
