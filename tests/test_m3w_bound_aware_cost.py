from copy import deepcopy
import numpy as np
import pytest
import torch

from src.world_model import m3w_bound_aware_cost as api


def example():
    rng=np.random.default_rng(8);n=12
    c=dict(x=rng.normal(size=(n,7,8)).astype(np.float32),envelope=np.full((n,7),3.),
        scales=np.ones(n,np.float32),allowed=np.ones((n,7),bool),groups=[np.array([i]) for i in range(n)],
        source_groups=[np.arange(6),np.arange(6,12)],sites=np.array(['a']*6+['b']*6),ids=np.arange(n))
    paths=rng.normal(size=(n,7,12,2));target=rng.normal(size=(n,12,2));valid=np.ones((n,12),bool)
    valid[2]=False;valid[8]=False
    return api.training_data(c,paths,target,valid,1,3.)


def test_only_unknown_supervision_changes_and_common_preprocessing_is_shared():
    d=example();known=d['observed_future']
    torch.testing.assert_close(d['truth_by_arm']['known_only'][known],d['truth_by_arm']['bound_aware'][known],rtol=0,atol=0)
    p=torch.ones((12,7,5));ix=np.arange(12);seg=torch.arange(12)
    a=api.objective(p,d,ix,seg,12,'known_only');b=api.objective(p,d,ix,seg,12,'bound_aware')
    assert a['unknown_bound_supervised_rows']==0 and b['unknown_bound_supervised_rows']==2
    assert a['observed_supervised_rows']==b['observed_supervised_rows']==10


def test_real_optimizer_resume_and_paired_draws_are_exact(tmp_path):
    torch.set_num_threads(1);d=example()
    settings=dict(width=8,learning_rate=.001,query_batch_size=4,minimum_updates=8,
        largest_source_cycles=2,gradient_clip=5.,heartbeat_every=2,checkpoint_every=2)
    states=[]
    for arm in api.ARMS:
        kw=dict(arm=arm,seed=17,settings=settings,identity={'synthetic':True})
        path=tmp_path/(arm+'.pt.gz')
        api.fit(d,path=path,stop_at=4,**kw)
        a=api.fit(d,path=path,resume=True,**kw)
        b=api.fit(d,path=tmp_path/(arm+'_direct.pt.gz'),**kw)
        for key in a:
            if key!='seconds':api.base.exact(a[key],b[key])
        states.append(a)
    for field in ('initial_model','parameters','budget','sampler','draw_hash','fit_ids','preprocess','torch_rng'):
        api.base.exact(states[0][field],states[1][field])


def test_bounded_decision_is_causal_and_obeys_its_predicted_moments():
    p=np.zeros((3,7,5));p[:,6]=[10,1,20,10,.5]
    p[1,6,0]=.5;legal=np.ones((3,7),bool);legal[2,6]=False
    a=api.amplitudes(p,legal)
    np.testing.assert_array_equal(a,[.4,0,0])
    assert a[0]*p[0,6,1]<=.02*p[0,6,2]
    assert a[0]*p[0,6,4]<=.02*p[0,6,3]
    assert not api.amplitudes(p,legal,binary=True).any()
    with pytest.raises(ValueError):api.amplitudes(p,legal,budget=.03)


def test_no_future_payload_is_accepted_by_existing_inference():
    c=deepcopy(example()['causal']);c['future_valid']=np.ones((12,12),bool)
    with pytest.raises(ValueError):api.base.predict({},c)
