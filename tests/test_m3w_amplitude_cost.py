import numpy as np
import pytest
import torch
from src.world_model import m3w_amplitude_cost as api
from src.world_model import m3w_shared_motion_cost as original
from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_supported_bound_targets import supported_bound_targets


def packet():
    t=np.arange(8.)
    history=np.stack([np.column_stack((t+j,2*t-j))for j in range(4)])
    g,ids,shared=encode_scene(history,np.ones((4,8),bool),np.arange(4))
    paths=np.stack([baseline_numpy(history,k)-history[:,-1,None]for k in range(6)],1)
    paths=np.concatenate((paths,(paths[:,1]+3)[:,None]),1)
    c=original.features(geometry=g,shared_context=shared,rollouts=paths,floor_index=1,
        recordings=np.array([0,0,1,1]),frames=np.full(4,7),ids=ids,sites=np.array(['s']*4))
    c['allowed'][:,:6]=False
    inputs=dict(old_causal=c,geometry=g,shared_context=shared,paths=paths,floor_index=1)
    target=paths[:,1].copy()
    target[1]+=20
    target[3]-=.7
    valid=np.ones((4,12),bool);valid[2]=False;target[2]=np.nan
    pr=dict(mean=np.zeros(428),std=np.ones(428),clip=8.,rms=np.ones(5,np.float32),bias=np.zeros(5,np.float32))
    return inputs,target,valid,pr


def test_causal_bank_full_amplitude_and_floor_are_exact_with_true_envelopes():
    inputs,_,_,_=packet()
    out=api.causal_bank(**inputs)
    np.testing.assert_array_equal(out['paths'][:,0],inputs['paths'][:,1])
    np.testing.assert_array_equal(out['paths'][:,6],inputs['paths'][:,6])
    np.testing.assert_array_equal(out['causal']['x'][:,6],inputs['old_causal']['x'][:,6])
    np.testing.assert_array_equal(out['cv'],inputs['paths'][:,1])
    assert set(out['causal']) == original.base.base.CAUSAL_KEYS
    assert not out['causal']['allowed'][:,0].any()
    np.testing.assert_allclose(out['causal']['envelope'],np.linalg.norm(out['paths']-out['paths'][:,0,None],axis=-1).max(-1))
    with pytest.raises(TypeError):
        api.causal_bank(**inputs,future_endpoint=np.zeros((4,2)))


def test_original_cv_not_first_intermediate_action_defines_easy_labels():
    inputs,target,valid,_=packet()
    out=api.causal_bank(**inputs)
    truth=api.targets(out,target,valid,1.)
    old=supported_bound_targets(inputs['paths'],target,valid,1,1.)
    np.testing.assert_array_equal(truth['observed_future'],old['observed_future'])
    np.testing.assert_array_equal(truth['moments'][:,6],old['moments'][:,6])
    assert not truth['moments'][0,:,3].any()  # exact CV fit is not a positive-CV-error easy event
    assert truth['moments'][3,:,3].min() > 0  # changing slot1must not change the CV-defined event
    for k in range(7):
        assert truth['moments'][2,k,2] == truth['moments'][2,k,3] == 0
        assert truth['moments'][2,k,1] == truth['moments'][2,k,4] == out['causal']['envelope'][2,k]


def test_matched_inputs_labels_normalization_and_only_action_mask_differs():
    inputs,target,valid,pr=packet()
    common=api.training_data(api.causal_bank(**inputs),target,valid,1.,pr)
    a,b=(api.for_arm(common,arm)for arm in api.ARMS)
    assert a['z'] is b['z'] and a['preprocess'] is b['preprocess']
    torch.testing.assert_close(a['truth_by_arm']['bound_aware'],b['truth_by_arm']['bound_aware'],rtol=0,atol=0)
    assert not torch.count_nonzero(a['z'][...,380:])
    assert not a['causal']['allowed'][:,:6].any()
    assert b['causal']['allowed'][:,1:].all()
    assert common['causal']['allowed'][:,1:].all()
    model=original.base.base.CostHead(428,64,17,pr['bias'],'context')
    assert sum(p.numel()for p in model.parameters()) == 28229


@pytest.mark.parametrize('arm',api.ARMS)
def test_fresh_fit_resume_and_prediction_are_exact_without_future_inputs(tmp_path,arm):
    inputs,target,valid,pr=packet()
    common=api.training_data(api.causal_bank(**inputs),target,valid,1.,pr)
    settings=dict(width=8,learning_rate=.001,query_batch_size=2,minimum_updates=4,largest_source_cycles=1,
        gradient_clip=5.,checkpoint_every=1,heartbeat_every=1)
    kwargs=dict(arm=arm,seed=17,settings=settings,identity=dict(synthetic=True))
    direct=api.fit(common,path=tmp_path/'direct.pt.gz',**kwargs)
    api.fit(common,path=tmp_path/'resume.pt.gz',stop_at=2,**kwargs)
    recovered=api.fit(common,path=tmp_path/'resume.pt.gz',resume=True,**kwargs)
    original.base.base.exact({k:v for k,v in direct.items()if k!='seconds'},
        {k:v for k,v in recovered.items()if k!='seconds'})
    np.testing.assert_array_equal(api.predict(direct,common['causal']),api.predict(recovered,common['causal']))
    binary=api.for_arm(common,'binary_only')['causal']
    np.testing.assert_array_equal(api.predict(direct,common['causal']),api.predict(direct,binary))
    with pytest.raises(ValueError):
        api.predict(direct,dict(common['causal'],future_valid=valid))
