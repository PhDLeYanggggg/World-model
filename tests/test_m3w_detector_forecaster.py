import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_detector_forecaster as api

ARCH=dict(width=16,heads=2,layers=1,neighbor_policy='complete_aligned_history',
    input_conditioning='observed_joint_max_norm',output_parameterization='motion_bounded')
SETTINGS=dict(steps=20,batch_size=4,learning_rate=.0003,minimum_lr_ratio=.01,weight_decay=.0001,
    gradient_clip=5.,checkpoint_every=10,heartbeat_every=10)


def data():
    rng=np.random.default_rng(22);n=16;g=np.zeros((n,476),np.float32)
    g[:,:16]=rng.normal(size=(n,16));g[:,16:24]=np.arange(-7,1)/12
    g[:,332:356]=rng.normal(size=(n,24));x=rng.normal(size=(n,64)).astype(np.float32)
    values=dict(geometry=g,target=rng.normal(size=(n,12,2)).astype(np.float32),valid=np.ones((n,12),bool),sites=np.array(['a']*8+['b']*8))
    fold=dict(train_ids=np.arange(8),held_ids=np.arange(8,16),groups=[np.arange(8)],factors=np.r_[np.ones(8),np.zeros(8)])
    return values,x,fold


def test_zero_detector_branch_reproduces_nontrivial_original_forward_and_strict_schema():
    torch.set_num_threads(1);d,x,_=data();torch.manual_seed(17)
    model=api.DetectorForecaster(ARCH,1,'history')
    with torch.no_grad():model.predictor.predictor.predictor.output[-1].weight.fill_(.02)
    inputs=api.ORIGINAL_PACK(d['geometry']);extended=dict(inputs,detector_features=torch.from_numpy(x.reshape(-1,8,8).transpose(0,2,1).copy()))
    model.eval()
    with torch.no_grad():
        a=api.SourceForecaster.forward(model,inputs);b=model(extended)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError):model(dict(extended,future_endpoint=torch.zeros(16,2)))


def test_disabled_forecaster_exact_old_training(tmp_path):
    d,x,fold=data();torch.manual_seed(17);old=api.SourceForecaster(ARCH,1)
    api.native.fit_trial(old,d,fold,seed=17,settings=SETTINGS,identity={},directory=tmp_path/'old',heartbeat=lambda **_:None)
    oldstate=torch.load(tmp_path/'old/checkpoint.pt',weights_only=False)
    padded=dict(d,geometry=api.DetectorGeometry(d['geometry'],x,np.zeros(64),np.ones(64)))
    torch.manual_seed(17);new=api.DetectorForecaster(ARCH,1,'disabled')
    with api.installed():api.native.fit_trial(new,padded,fold,seed=17,settings=SETTINGS,identity={},directory=tmp_path/'new',heartbeat=lambda **_:None)
    api.assert_neutral(oldstate,torch.load(tmp_path/'new/checkpoint.pt',weights_only=False))


def test_active_producer_branch_updates_and_recovers_exactly(tmp_path):
    d,x,fold=data();d=dict(d,geometry=api.DetectorGeometry(d['geometry'],x,np.zeros(64),np.ones(64)))
    def fit(folder,stop=None,resume=False):
        torch.manual_seed(17);model=api.DetectorForecaster(ARCH,1,'history')
        with api.installed():api.native.fit_trial(model,d,fold,seed=17,settings=SETTINGS,identity={},directory=folder,heartbeat=lambda **_:None,stop_at=stop,resume=resume)
        return torch.load(folder/'checkpoint.pt',weights_only=False)
    fit(tmp_path/'resumed',10);a=fit(tmp_path/'resumed',resume=True);b=fit(tmp_path/'direct')
    api.assert_recovery(a,b);assert torch.count_nonzero(a['model']['detector_weight'])>0
    assert not a['draws'][fold['held_ids']].any()
    with api.installed():
        with pytest.raises(ValueError):api.pack_geometry(np.zeros((1,476)))
    assert api.native.pack_geometry is api.ORIGINAL_PACK
