import numpy as np

from src.world_model import m3w_one_slot_reservation as model


def test_only_earned_slot_can_move_and_price_is_past_opportunity_cost():
    groups = [np.arange(i,i+3) for i in (0,3,6)]
    mint = np.array([0,0,-1,-1,-1,-1,-1,-1,-1])
    u = np.array([[4.],[2.],[1.],[1.],[.5],[.2],[3.],[2.5],[.1]])
    zero = np.zeros_like(u)
    legal = np.ones_like(u,bool)
    q,r = model.allocate(groups,u,zero,zero,legal,mint)
    np.testing.assert_array_equal(q,[0,-1,-1,-1,-1,-1,0,-1,-1])
    assert r==dict(held=1,released=1,unused=0)
    prev,p = model.allocate(groups[:2],u[:6],zero[:6],zero[:6],legal[:6],mint[:6])
    np.testing.assert_array_equal(prev,q[:6])
    assert p==dict(held=1,released=0,unused=1)
