import numpy as np
import pytest
import torch
from src.world_model.m3w_european_source_forecast import (
    baseline_numpy,baseline_torch,source_folds,pack_scene,fit_design,SourceForecaster)
from src.world_model.m3w_native_forecast import pack_geometry,masked_objective
from src.world_model.m3w_native_forecast import fit_trial


def test_baseline_arithmetic_and_causal_constant_slope():
    h = np.arange(8)[None,:,None]*np.array([[[2.,3.]]])+5
    for k in range(6):
        np.testing.assert_allclose(baseline_numpy(h,k),baseline_torch(torch.tensor(h),k).numpy(),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(baseline_numpy(h,1),baseline_numpy(h,4))
    np.testing.assert_allclose(baseline_numpy(h,1),baseline_numpy(h,5))
    h2 = h+np.array([[[17.,-3.]]])
    np.testing.assert_allclose(baseline_numpy(h2,4)-baseline_numpy(h,4),np.broadcast_to([17.,-3.],(1,12,2)))


def test_source_groups_balanced_stable():
    s = {str(i):i*50 for i in range(12)}
    f = source_folds(s,'fixed')
    assert f == source_folds(dict(reversed(list(s.items()))),'fixed')
    assert [list(f.values()).count(k) for k in range(3)] == [4,4,4]
    with pytest.raises(ValueError):
        source_folds({'x':1},'fixed')


def input_scene():
    h = np.tile(np.arange(8)[None,:,None],(3,1,2)).astype(float)
    h[1] += 10
    h[2] += 20
    valid = np.ones((3,8),bool)
    valid[2,:4] = False
    return dict(history_xy=h,history_valid=valid,target_eligible=valid.all(1),
                agent_id=np.array([1,2,3]),baseline_cv=baseline_numpy(h,1))


def test_packing_uses_only_complete_past_neighbors_and_refuses_future():
    x = input_scene()
    g,ids = pack_scene(x)
    assert ids.tolist() == [0,1]
    p = pack_geometry(g)
    assert p['neighbor_mask'].sum().item() == 16
    assert not (p['history'][...,2]>0).any()
    assert torch.equal(p['history'][:,-1,:2],torch.zeros(2,2))
    with pytest.raises(ValueError):
        pack_scene({**x,'future_xy':np.ones((3,12,2))})
    shifted = dict(x,history_xy=x['history_xy']+100,baseline_cv=x['baseline_cv']+100)
    np.testing.assert_array_equal(g,pack_scene(shifted)[0])


def test_training_design_never_uses_held_costs():
    sites = np.repeat(['a','b','c'],3)
    costs = np.tile([3.,2.,1.,2.,2.,2.],(9,1))
    costs[0] = np.nan
    data = {'sites':sites}
    a = fit_design(data,['a','b'],costs)
    changed = costs.copy()
    changed[sites=='c'] = 1e9
    b = fit_design(data,['a','b'],changed)
    assert a['baseline_index']==b['baseline_index']==2
    np.testing.assert_array_equal(a['factors'],b['factors'])
    assert not a['factors'][sites=='c'].any()
    assert a['normalizers']['a']['correction']==1.5


def test_real_torch_gradients_and_initial_strong_baseline():
    torch.set_num_threads(4)
    p = pack_geometry(pack_scene(input_scene())[0])
    cfg = dict(width=16,heads=2,layers=1,neighbor_policy='complete_aligned_history',
               input_conditioning='observed_joint_max_norm',output_parameterization='motion_bounded')
    model = SourceForecaster(cfg,4)
    prediction = model(p)
    torch.testing.assert_close(prediction,baseline_torch(p['history'],4),rtol=0,atol=0)
    loss,_ = masked_objective(prediction,prediction.detach()+1,torch.ones(2,12,dtype=torch.bool),torch.ones(2))
    loss.backward()
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in model.parameters())


def test_checkpoint_resume_preserves_exact_weights_and_draws(tmp_path):
    g,_ = pack_scene(input_scene())
    geometry = np.tile(g,(6,1))
    data = dict(geometry=geometry,target=geometry[:,332:356].reshape(-1,12,2).copy()+1,
                valid=np.ones((12,12),bool),sites=np.repeat(['a','b','c'],4))
    costs = np.tile([3.,2.,1.,2.,2.,2.],(12,1))
    fold = fit_design(data,['a','b'],costs)
    cfg = dict(width=16,heads=2,layers=1,neighbor_policy='complete_aligned_history',
               input_conditioning='observed_joint_max_norm',output_parameterization='motion_bounded')
    settings = dict(steps=6,batch_size=4,learning_rate=.001,minimum_lr_ratio=.01,
                    weight_decay=.0001,gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    def train(model,path,**kwargs):
        return fit_trial(model,data,fold,seed=17,settings=settings,identity={'synthetic':True},
                         directory=path,heartbeat=lambda **kw:None,**kwargs)
    torch.manual_seed(17)
    full = SourceForecaster(cfg,2)
    train(full,tmp_path/'full')
    torch.manual_seed(17)
    interrupted = SourceForecaster(cfg,2)
    train(interrupted,tmp_path/'split',stop_at=3)
    resumed = SourceForecaster(cfg,2)
    train(resumed,tmp_path/'split',resume=True)
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
    a = torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b = torch.load(tmp_path/'split/checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(a['draws'],b['draws'])
    assert a['draws'][fold['held_ids']].sum() == 0
