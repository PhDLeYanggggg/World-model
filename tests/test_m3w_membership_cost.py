import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_membership_cost as m
from src.world_model.m3w_native_gain_harm import preprocess


def fixture():
    rng = np.random.default_rng(9); x = rng.normal(size=(30, 8))
    cv = np.linspace(.05, 3, 30); cv[7] = np.nan; sites = np.repeat(['a','b','c'], 10)
    pr = preprocess(x, np.column_stack((cv, cv)), cv, sites, 'held')
    easy = np.where(np.isfinite(cv), ((cv>0)&(cv<=pr['positive_easy_cut'])).astype(float), np.nan)
    env = np.linspace(.3, 2, 30); h = env*.2; h[::3] = 0
    y = np.column_stack((cv, h, cv*easy, h*easy)); y[~np.isfinite(cv)] = np.nan
    cfg = dict(steps=8, width=4, batch_size=9, learning_rate=.001, gradient_clip=5, heartbeat_every=2, checkpoint_every=2)
    return x, y, easy, sites, env, pr, cfg


@pytest.mark.parametrize('arm', ['direct','conditional'])
def test_exact_resume_and_unknown_exclusion(tmp_path, arm):
    torch.set_num_threads(1); x,y,e,s,d,p,c = fixture()
    kw = dict(arm=arm, seed=17, settings=c, identity={'test':1}, heartbeat=lambda **_:None)
    a,r = m.fit(x,y,e,s,d,p,directory=tmp_path/'a',**kw)
    m.fit(x,y,e,s,d,p,directory=tmp_path/'b',stop_at=4,**kw)
    b,_ = m.fit(x,y,e,s,d,p,directory=tmp_path/'b',resume=True,**kw)
    pk = dict(arm=arm, probability=np.full(len(x), .3))
    np.testing.assert_array_equal(m.predict(a,x,d,p,**pk), m.predict(b,x,d,p,**pk))
    _,u=m.restore(tmp_path/'a'); _,v=m.restore(tmp_path/'b')
    np.testing.assert_array_equal(u['draws'],v['draws']); assert u['draws'][7]==0 and r['unknown_rows_sampled']==0
    assert torch.equal(u['sampler_rng'],v['sampler_rng'])
    with pytest.raises(ValueError): m.fit(x,y,e,s,d,p,directory=tmp_path/'a',resume=True,**dict(kw,identity={'test':2}))


def test_sampling_matches_and_membership_is_not_fitting_input(tmp_path):
    assert 'probability' not in inspect.signature(m.fit).parameters
    assert 'easy' not in inspect.signature(m.predict).parameters
    x,y,e,s,d,p,c = fixture(); states=[]
    for arm in ('direct','conditional'):
        model,_=m.fit(x,y,e,s,d,p,arm=arm,seed=29,settings=c,identity={},directory=tmp_path/arm,heartbeat=lambda **_:None)
        states.append(m.restore(tmp_path/arm)[1])
        out=m.predict(model,x,d,p,arm=arm,probability=np.linspace(0,1,len(x)))
        assert np.all(out[:,1]<=out[:,0]) and np.all(out[:,0]<=d+1e-5)
    np.testing.assert_array_equal(states[0]['draws'],states[1]['draws'])
    np.testing.assert_array_equal(states[0]['fixed_ids'],states[1]['fixed_ids'])


def test_conditional_composition_includes_zero_harm_easy():
    v=torch.tensor([[0.,2.],[1.,3.]])
    out=m.moments(v,'conditional',torch.tensor([1.,.25]))
    torch.testing.assert_close(out,torch.tensor([[0.,0.],[2.5,.25]]))


def test_loss_uses_conditional_masks_not_predicted_membership():
    model=m.CostHead(1,2)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    z=torch.zeros(2,1); d=torch.ones(2); h=torch.tensor([0.,1.]); e=torch.tensor([1.,0.])
    got=m.objective(model,z,d,h,e,'conditional',torch.ones(2),.5)
    assert float(got.detach())==pytest.approx(.25)


def test_invalid_supervision_and_probability_rejected(tmp_path):
    x,y,e,s,d,p,c=fixture(); bad=e.copy(); bad[0]=1-bad[0]
    y[0,1]=.1; y[0,3]=.1*e[0]
    with pytest.raises(AssertionError): m.fit(x,y,bad,s,d,p,arm='conditional',seed=1,settings=c,identity={},directory=tmp_path,heartbeat=lambda **_:None)
    with pytest.raises(ValueError): m.predict(m.CostHead(8,4),x,d,p,arm='conditional',probability=np.full(len(x),1.1))
