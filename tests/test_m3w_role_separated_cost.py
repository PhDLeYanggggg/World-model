import inspect
import numpy as np
import torch

from src.world_model import m3w_role_separated_cost as m


def data():
    rng=np.random.default_rng(17);n=90
    y=rng.uniform(.01,.5,(n,5)).astype(np.float32);y[:,3]=y[:,2]*.5;y[:,4]=y[:,1]*.5
    known=np.arange(n)%5!=0;y[~known]=[0,1,0,0,1]
    return dict(x=rng.normal(size=(n,120)),moments=y,observed=known,envelope=np.ones(n),
                sites=np.repeat(['a','b','c'],30),weights=np.ones(n)/n,ids=np.arange(n))


def test_held_future_features_and_roles_do_not_change_fitting():
    d=data();a=m.prepare(d,'a')
    d['moments'][:30]=1e6;d['x'][:30]=1e6;d['observed'][:30]=~d['observed'][:30]
    b=m.prepare(d,'a');m.storage.storage.base.exact(a,b)


def test_unknown_magnitude_does_not_move_known_normalizer():
    d=data();a=m.prepare(d,'a');d['moments'][~d['observed']]*=1000
    b=m.prepare(d,'a')
    for k in ('mean','std','rms','bias','observation_prior'):
        np.testing.assert_array_equal(a['binding'][k],b['binding'][k])


def test_conditional_gradient_isolated_from_unknown_targets_and_probability():
    p=m.prepare(data(),'a');model=m.Head(m.SETTINGS,p['binding'])
    out=model(p['x'],p['envelope']);terms=m.objective(out,p['y'],p['known'],p['rms'],'role_separated')
    first=torch.autograd.grad(terms['observed_conditional_MSE'],tuple(model.parameters()),allow_unused=True,retain_graph=True)
    assert all(v is None for v in first[-4:])
    y=p['y'].clone();y[~p['known']]*=1000
    second=torch.autograd.grad(m.objective(out,y,p['known'],p['rms'],'role_separated')['observed_conditional_MSE'],
                               tuple(model.parameters()),allow_unused=True)
    for a,b in zip(first,second):
        if a is None:assert b is None
        else:torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_composed_moments_are_coherent_and_probability_is_action_invariant():
    p=m.prepare(data(),'a');model=m.Head(m.SETTINGS,p['binding'])
    x=p['x'][:3].clone();x[1:,:50]=x[0,:50]
    out=model(x,torch.ones(3))
    assert torch.equal(out['observation_probability'],out['observation_probability'][0].expand(3))
    v=out['moments'];assert (v>=0).all() and (v[:,0]+v[:,1]<=1+1e-6).all()
    assert (v[:,0]<=v[:,2]).all() and (v[:,3]<=v[:,2]).all() and (v[:,4]<=v[:,1]).all()
    assert not {'valid','observed','target','future_endpoint'}&set(inspect.signature(m.predict).parameters)


def test_actual_optimizer_resume_matches_uninterrupted(tmp_path):
    d=data();m.fit(d,'a',{},tmp_path/'r.gz',arm='role_separated',seed=29,stop_at=2)
    resumed=m.fit(d,'a',{},tmp_path/'r.gz',arm='role_separated',seed=29,resume=True,stop_at=4)
    direct=m.fit(d,'a',{},tmp_path/'d.gz',arm='role_separated',seed=29,stop_at=4)
    m.exact(resumed,direct)


def test_matched_arms_have_same_initial_weights_and_draws(tmp_path):
    d=data()
    a=m.fit(d,'a',{},tmp_path/'a.gz',arm='role_separated',seed=17,stop_at=4)
    b=m.fit(d,'a',{},tmp_path/'b.gz',arm='composed_anchor',seed=17,stop_at=4)
    for k in ('preprocess','initial_model','generator','visits','parameters','step'):
        m.storage.storage.base.exact(a[k],b[k])
