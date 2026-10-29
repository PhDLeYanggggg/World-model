import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_multiaction_forest as api
from src.world_model import m3w_multiaction_cost as base
from tests.test_m3w_scene_context_cost import scene_sample,SETTINGS
from src.evaluation.m3w_multiaction_forest_readout import subsets,pooled_subsets

TREE=dict(trees=4,max_depth=4,min_samples_leaf=1,max_features=1.,fit_threads=1,checkpoint_every=2)


def test_draw_replay_known_mask_and_query_averaged_weights(tmp_path):
    torch.set_num_threads(1);*_,data=scene_sample()
    counts,digest=api.replay_draws(data,SETTINGS,17)
    assert counts.sum()==SETTINGS['steps']*SETTINGS['batch_size'] and counts[-1]==0
    model=base.fit(data,arm='physical_neural',settings=SETTINGS,seed=17,identity={},path=tmp_path/'torch.gz')
    assert digest==model['draw_hash']
    data['allowed'][:,2]=False
    matrix=api.training_matrix(data,'physical_only',counts)
    expected=sum(counts[i] for i in range(len(counts)) if data['allowed'][i,:6].any())
    assert np.isclose(matrix['weight'].sum()*matrix['query_weight_rescale'],expected)
    assert matrix['x'].shape[1]==data['z'].shape[-1]+8


def test_all_bank_resume_exact_and_inactive_neural_does_not_change_physical(tmp_path):
    torch.set_num_threads(1);*_,data=scene_sample();counts,digest=api.replay_draws(data,SETTINGS,17)
    original=None
    for arm in base.ARMS:
        matrix=api.training_matrix(data,arm,counts)
        kwargs=dict(preprocess=data['preprocess'],arm=arm,settings=TREE,seed=17,identity={},draw_hash=digest)
        direct=api.fit(matrix,**kwargs,path=tmp_path/(arm+'.joblib'))
        api.fit(matrix,**kwargs,path=tmp_path/(arm+'_resume.joblib'),stop_at=1)
        resumed=api.fit(matrix,**kwargs,path=tmp_path/(arm+'_resume.joblib'),resume=True)
        api.exact_forest(direct,resumed)
        if arm=='physical_only':original=direct
    altered=copy.deepcopy(data);altered['z'].motion[:,6]+=1e5;altered['z'].scene[:,6]+=1e5;altered['truth'][:,6]+=1e5
    other=api.fit(api.training_matrix(altered,'physical_only',counts),preprocess=data['preprocess'],arm='physical_only',
        settings=TREE,seed=17,identity={},draw_hash=digest,path=tmp_path/'unused.joblib')
    api.exact_forest(original,other)


def test_inference_projects_causal_bounds_and_cannot_accept_future_endpoints(tmp_path):
    torch.set_num_threads(1);x,scene,envelope,data=scene_sample();counts,digest=api.replay_draws(data,SETTINGS,17)
    state=api.fit(api.training_matrix(data,'physical_neural',counts),preprocess=data['preprocess'],arm='physical_neural',
        settings=TREE,seed=17,identity={},draw_hash=digest,path=tmp_path/'model.joblib')
    p=api.predict(state,x,scene,envelope,np.ones(len(x)))
    assert p.shape==(len(x),7,5) and np.isfinite(p).all() and (p>=0).all()
    assert (p[:,:,:2].sum(-1)<=envelope+1e-6).all() and (p[:,:,4]<=p[:,:,1]+1e-9).all()
    with pytest.raises(TypeError):api.predict(state,x,scene,envelope,np.ones(len(x)),future_endpoint=x[:,-1,:2])


def test_resume_rejects_changed_sampled_targets(tmp_path):
    torch.set_num_threads(1);*_,data=scene_sample();counts,digest=api.replay_draws(data,SETTINGS,17)
    matrix=api.training_matrix(data,'physical_neural',counts)
    kwargs=dict(preprocess=data['preprocess'],arm='physical_neural',settings=TREE,seed=17,
        identity={},draw_hash=digest,path=tmp_path/'resume.joblib')
    api.fit(matrix,**kwargs,stop_at=2)
    data['truth'][np.flatnonzero(counts)[0],0,0]+=1
    with pytest.raises(AssertionError):
        api.fit(api.training_matrix(data,'physical_neural',counts),**kwargs,resume=True)


def test_subset_unknown_and_empty_risk_never_fabricated():
    packet=dict(y=np.ones((3,7,5)),hard=np.array([True,False,False]),cv_ade=np.array([5.,1.,.1]),
        floor_ade=np.array([2.,1.,np.nan]),floor_fde=np.array([4.,2.,np.nan]),
        ade=np.tile([1.,.5,np.nan],(7,1)).T,fde=np.tile([2.,1.,np.nan],(7,1)).T)
    packet['y'][-1]=np.nan
    choice=dict(floor=np.full(3,-1),switch=np.zeros(3,dtype=int))
    value=subsets(packet,choice,1.)
    assert value['easy']['known_rows']==1 and value['hard']['known_rows']==1
    assert value['hard']['policies']['switch']['ADE_gain_percent']==50.
    packet['hard'][:]=False
    empty=subsets(packet,choice,1.)
    result=pooled_subsets([dict(result=dict(subsets=value)),dict(result=dict(subsets=empty))])
    assert result['hard']['empty_known_views']==1 and result['hard']['macro_status']=='not_estimable'
    assert empty['hard']['policies']['switch']['ADE_gain_percent'] is None
