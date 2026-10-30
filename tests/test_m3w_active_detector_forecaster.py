import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_active_detector_forecaster as api

SETTINGS=dict(steps=20,batch_size=4,learning_rate=.0003,minimum_lr_ratio=.01,weight_decay=.0001,
    gradient_clip=5.,checkpoint_every=10,heartbeat_every=10)


def data():
    rng=np.random.default_rng(25);n=16;g=np.zeros((n,476),np.float32)
    g[:,:16]=rng.normal(size=(n,16));g[:,16:24]=np.arange(-7,1)/12
    g[:,38:166]=rng.normal(size=(n,128));g[:,166:230]=np.tile(np.arange(-7,1)/12,8)
    g[:,230:294]=rng.integers(0,2,size=(n,64));extra=rng.normal(size=(n,64)).astype(np.float32)
    d=dict(geometry=g,target=rng.normal(size=(n,12,2)).astype(np.float32),valid=np.ones((n,12),bool),sites=np.array(['a']*8+['b']*8))
    fold=dict(train_ids=np.arange(8),held_ids=np.arange(8,16),groups=[np.arange(8)],factors=np.r_[np.ones(8),np.zeros(8)])
    return d,extra,fold


def test_zero_input_branch_retains_nontrivial_active_topology_masks_and_units():
    torch.set_num_threads(1);d,x,_=data();torch.manual_seed(17)
    model=api.ActiveDetectorForecaster(1,'history',width=16,heads=2)
    with torch.no_grad():model.predictor.predictor.predictor.output[-1].weight.fill_(.02)
    plain=api.ORIGINAL_PACK(d['geometry']);inputs=dict(plain,detector_features=torch.from_numpy(x.reshape(-1,8,8).transpose(0,2,1).copy()))
    model.eval()
    with torch.no_grad():
        a=api.DimensionlessAgentTrackSourceForecaster.forward(model,plain);b=model(inputs)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        padded=copy.deepcopy(inputs);padded['neighbors'][~padded['neighbor_mask']]=1e8
        torch.testing.assert_close(model(padded),b,rtol=0,atol=0)
    with pytest.raises(ValueError):model(dict(inputs,future_endpoint=torch.zeros(16,2)))
    broken=copy.deepcopy(inputs);broken['prediction_time'][:,0]=-1
    with pytest.raises(ValueError):model(broken)
    assert not model.predictor.predictor.predictor.embed._forward_hooks


def test_disabled_branch_exact_active_parent_training(tmp_path):
    d,x,fold=data();torch.manual_seed(17)
    old=api.DimensionlessAgentTrackSourceForecaster(1,width=16,heads=2)
    api.native.fit_trial(old,d,fold,seed=17,settings=SETTINGS,identity={},directory=tmp_path/'old',heartbeat=lambda **_:None)
    prepared=dict(d,geometry=api.DetectorGeometry(d['geometry'],x,np.zeros(64),np.ones(64)))
    torch.manual_seed(17);new=api.ActiveDetectorForecaster(1,'disabled',width=16,heads=2)
    with api.installed():api.native.fit_trial(new,prepared,fold,seed=17,settings=SETTINGS,identity={},directory=tmp_path/'new',heartbeat=lambda **_:None)
    api.assert_neutral(torch.load(tmp_path/'old/checkpoint.pt',weights_only=False),torch.load(tmp_path/'new/checkpoint.pt',weights_only=False))


def test_actual_active_updates_and_exact_resume_keep_held_draws_zero(tmp_path):
    d,x,fold=data();d=dict(d,geometry=api.DetectorGeometry(d['geometry'],x,np.zeros(64),np.ones(64)))
    def fit(path,stop=None,resume=False):
        torch.manual_seed(17);model=api.ActiveDetectorForecaster(1,'history',width=16,heads=2)
        with api.installed():api.native.fit_trial(model,d,fold,seed=17,settings=SETTINGS,identity={},directory=path,
            stop_at=stop,resume=resume,heartbeat=lambda **_:None)
        return torch.load(path/'checkpoint.pt',weights_only=False)
    fit(tmp_path/'resumed',10);a=fit(tmp_path/'resumed',resume=True);b=fit(tmp_path/'direct')
    api.assert_recovery(a,b)
    assert torch.count_nonzero(a['model']['detector_weight'])>0 and not a['draws'][fold['held_ids']].any()
