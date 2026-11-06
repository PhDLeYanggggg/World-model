import numpy as np
from src.data_unification import m3w_full_action_features as m
from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model.m3w_european_source_forecast import baseline_numpy


def test_all_actions_retain_the_original_proposal_basis():
    h=np.stack([np.column_stack((np.arange(8)+i*10,np.arange(8)*.5)) for i in range(3)])
    geometry,_,context=encode_scene(h,np.ones((3,8),bool),np.arange(3))
    forecast=baseline_numpy(h,1)-h[:,-1,None]+np.arange(1,13)[None,:,None]*np.array([.2,.1])
    p=m.packet.packet(geometry=geometry,shared_context=context,history=h,recordings=np.zeros(3,int),
        frames=np.ones(3,int),ids=np.arange(3),sites=np.repeat('a',3),relative_forecast=forecast,floor_index=1)
    all_x=m.summary(p['causal']);proposal=np.array([1,-1,6]);active,x=m.packet.summary(p['causal'],proposal)
    np.testing.assert_array_equal(all_x[active,proposal[active]],x)
    assert all_x.shape==(3,7,120)
