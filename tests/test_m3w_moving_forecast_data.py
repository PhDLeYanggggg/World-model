import numpy as np

from src.data_unification.m3w_moving_forecast_data import encode_scene,SHARED_NAMES
from src.evaluation.m3w_dronecrowd_windows import Recording,past_inputs,WindowSpec


def test_partial_past_agent_stays_in_shared_context_but_not_complete_past_neighbor_control():
    h=np.zeros((3,8,2));h[0,:,0]=np.arange(8);h[1,:,1]=np.arange(8)
    h[2,:,0]=2*np.arange(8)+3
    mask=np.ones((3,8),bool);mask[2,:4]=False
    g,local,shared=encode_scene(h,mask,np.arange(3))
    assert list(local)==[0,1] and g.shape==(2,476) and shared.shape==(2,len(SHARED_NAMES))
    assert g[0,230:294].reshape(8,8).sum()==8
    assert shared[0,42]==np.float32(np.log1p(3)) and shared[0,43]==np.float32(2/3)
    _,_,without=encode_scene(h[:2],mask[:2],np.arange(2))
    assert not np.array_equal(shared,without)


def test_new_geometry_and_shared_features_ignore_changed_or_truncated_future():
    pos=np.zeros((3,24,2));pos[:,:,0]=np.arange(24)[None]+np.arange(3)[:,None]
    valid=np.ones((3,24),bool);valid[2,:6]=False
    rec=Recording('synthetic','audit_only',np.arange(3),pos,valid,np.zeros_like(valid))
    spec=WindowSpec('native8_12',8,1,tuple(range(1,13)));before=past_inputs(rec,9,spec)
    expected=encode_scene(before['positions'],before['history_mask'],before['agent_ids'])
    changed=pos.copy();changed[:,10:]+=10000;v=valid.copy();v[:,10:]=False
    other=Recording('synthetic','not_a_feature',np.arange(3),changed,v,np.zeros_like(v))
    after=past_inputs(other,9,spec)
    actual=encode_scene(after['positions'],after['history_mask'],after['agent_ids'])
    truncated=Recording('synthetic','not_a_feature',np.arange(3),pos[:,:10],valid[:,:10],np.zeros_like(valid[:,:10]))
    short=past_inputs(truncated,9,spec);again=encode_scene(short['positions'],short['history_mask'],short['agent_ids'])
    for a,b,c in zip(expected,actual,again):
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
