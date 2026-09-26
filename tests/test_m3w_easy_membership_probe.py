import numpy as np
import pytest
import torch
from src.world_model import m3w_easy_membership_probe as m
from src.world_model.m3w_native_gain_harm import preprocess


def fixture():
    rng=np.random.default_rng(9); x=rng.normal(size=(24,8)); cv=np.linspace(0,3,24); cv[5]=np.nan
    s=np.repeat(['a','b','c'],8); pr=preprocess(x,np.column_stack((cv,cv)),cv,s,'held')
    y=m.labels(cv,pr['positive_easy_cut'])
    settings=dict(steps=8,width=4,batch_size=6,learning_rate=.001,gradient_clip=5,heartbeat_every=2,checkpoint_every=2)
    return x,y,s,pr,settings


def test_easy_membership_includes_harmless_easy_and_excludes_zero_cost():
    y=m.labels(np.array([0,.1,.5,.6,np.nan]),.5)
    np.testing.assert_equal(y,[0,1,1,0,np.nan])
    for bad in ([-1,0],[np.inf,0]):
        with pytest.raises(ValueError): m.labels(bad,.5)


@pytest.mark.parametrize('arm',['linear','mlp'])
def test_exact_resume_and_unknown_support(tmp_path,arm):
    torch.set_num_threads(1); x,y,s,p,c=fixture()
    kw=dict(arm=arm,seed=17,settings=c,identity={'test':1},heartbeat=lambda **_:None)
    a,fit=m.fit(x,y,s,p,directory=tmp_path/'a',**kw)
    m.fit(x,y,s,p,directory=tmp_path/'b',stop_at=4,**kw)
    b,_=m.fit(x,y,s,p,directory=tmp_path/'b',resume=True,**kw)
    np.testing.assert_array_equal(m.predict(a,x,p),m.predict(b,x,p))
    _,u=m.restore(tmp_path/'a'); _,v=m.restore(tmp_path/'b')
    np.testing.assert_array_equal(u['draws'],v['draws']); assert u['draws'][5]==0 and fit['unknown_rows_sampled']==0
    assert torch.equal(u['sampler_rng'],v['sampler_rng'])
    with pytest.raises(ValueError): m.fit(x,y,s,p,directory=tmp_path/'a',resume=True,**dict(kw,identity={'test':2}))


def test_identical_initial_outputs_and_matched_sampling(tmp_path):
    x,y,s,p,c=fixture(); traces=[]; states=[]
    for arm in ('linear','mlp'):
        _,r=m.fit(x,y,s,p,arm=arm,seed=29,settings=c,identity={},directory=tmp_path/arm,heartbeat=lambda **_:None)
        traces.append(r['trace'][0]); states.append(m.restore(tmp_path/arm)[1])
    assert traces[0]==traces[1]
    np.testing.assert_array_equal(states[0]['draws'],states[1]['draws'])


def test_prediction_does_not_consume_changed_labels(tmp_path):
    x,y,s,p,c=fixture(); model,_=m.fit(x,y,s,p,arm='mlp',seed=1,settings=c,identity={},directory=tmp_path,heartbeat=lambda **_:None)
    a=m.predict(model,x,p); y[:]=1-y
    np.testing.assert_array_equal(a,m.predict(model,x,p)); assert np.all((a>=0)&(a<=1))
