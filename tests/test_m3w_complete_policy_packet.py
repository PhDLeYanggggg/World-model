import inspect

import numpy as np

from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.data_unification import m3w_complete_policy_packet as p
from src.world_model.m3w_donor_augmentation import causal_features
from src.world_model.m3w_european_source_forecast import baseline_numpy


def test_full_amplitude_matches_original_raw120_without_any_labels():
    h=np.stack([np.column_stack((np.arange(8)+10*i,np.arange(8)*.3)) for i in range(3)])
    g,_,shared=encode_scene(h,np.ones((3,8),bool),np.arange(3))
    physical=np.stack([baseline_numpy(h,k)-h[:,-1,None] for k in range(6)],1)
    forecast=physical[:,1]+np.arange(1,13)[None,:,None]*np.array([.1,.2])
    ids,rec,frame,sites=np.arange(3),np.zeros(3,int),np.ones(3,int),np.array(['source']*3)
    packet=p.packet(geometry=g,shared_context=shared,history=h,recordings=rec,frames=frame,
                    ids=ids,sites=sites,relative_forecast=forecast,floor_index=1)
    active,x=p.summary(packet['causal'],np.full(3,6))
    old,_,_,_=causal_features(g,shared,physical,forecast,rec,frame,ids,sites,1)
    np.testing.assert_array_equal(active,ids)
    np.testing.assert_array_equal(x,old)
    assert not {'target','valid','future_endpoint'} & set(inspect.signature(p.packet).parameters)
