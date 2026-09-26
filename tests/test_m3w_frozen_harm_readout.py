import numpy as np
import pytest
import torch
from src.world_model import m3w_frozen_harm_readout as m
from src.world_model.m3w_selected_risk_learning import EventMomentHead


def fixture():
    rng=np.random.default_rng(2); h=rng.normal(size=(24,8)); env=np.linspace(.1,2,24); env[0]=0
    y=np.column_stack((np.ones(24),env*.2,np.ones(24)*.3,env*.05)); y[5]=np.nan
    known=np.isfinite(y).all(1); w=known/known.sum()
    pr=m.preprocess(h,dict(weights=w,known=known,cost_scale=1.,training_sites=['a','b','c']))
    settings=dict(steps=8,batch_size=6,learning_rate=.001,gradient_clip=5,heartbeat_every=2,checkpoint_every=2)
    return h,y,np.repeat(['a','b','c'],8),env,pr,settings


def test_bounds_and_reference_preservation():
    h,y,s,e,p,c=fixture(); model=m.HarmReadout(8); ref=np.zeros((24,4)); ref[:,0]=3; ref[:,2]=1
    got=m.predict(model,h,e,p,ref)
    assert np.array_equal(got[:,[0,2]],ref[:,[0,2]]) and np.all(got[:,3]<=got[:,1])
    assert np.all(got[:,1]<=e+1e-6) and got[0,1]==got[0,3]==0


def test_exact_resume_sampling_and_missing_labels(tmp_path):
    torch.set_num_threads(1)
    h,y,s,e,p,c=fixture()
    kwargs=dict(seed=17,settings=c,identity={'test':1},heartbeat=lambda **_:None)
    a,fit=m.fit(h,y,s,e,p,directory=tmp_path/'a',**kwargs)
    m.fit(h,y,s,e,p,directory=tmp_path/'b',stop_at=4,**kwargs)
    b,_=m.fit(h,y,s,e,p,directory=tmp_path/'b',resume=True,**kwargs)
    for k,v in a.state_dict().items(): assert torch.equal(v,b.state_dict()[k])
    _,x=m.restore(tmp_path/'a'); _,z=m.restore(tmp_path/'b')
    np.testing.assert_array_equal(x['draws'],z['draws']); assert x['draws'][5]==0
    assert torch.equal(x['sampler_rng'],z['sampler_rng']) and fit['unknown_rows_sampled']==0
    with pytest.raises(ValueError): m.fit(h,y,s,e,p,directory=tmp_path/'a',**dict(kwargs,identity={'test':2}),resume=True)


def test_frozen_encoder_has_no_gradients_or_parameter_changes():
    h,y,s,e,p,c=fixture(); encoder=EventMomentHead(8,4)
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    value=m.extract(encoder,h,p)
    assert value.shape==(24,4)
    assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in before.items())
    assert all(v.grad is None for v in encoder.parameters())


def test_fitting_norm_ignores_unknown_rows_and_held_rows():
    h,y,s,e,p,c=fixture(); source={k:p[k] for k in ('weights','known','cost_scale','training_sites')}
    a=m.preprocess(h,source); h[5]=1e7; b=m.preprocess(h,source)
    np.testing.assert_array_equal(a['mean'],b['mean']); np.testing.assert_array_equal(a['std'],b['std'])


def test_identical_constant_initialization_for_different_latents(tmp_path):
    h,y,s,e,p,c=fixture(); q=m.preprocess(h*3+2,{k:p[k] for k in ('weights','known','cost_scale','training_sites')})
    kw=dict(seed=29,settings=c,identity={'test':2},heartbeat=lambda **_:None,stop_at=1)
    _,a=m.fit(h,y,s,e,p,directory=tmp_path/'a',**kw)
    _,b=m.fit(h*3+2,y,s,e,q,directory=tmp_path/'b',**kw)
    assert a['trace'][0]==b['trace'][0]
