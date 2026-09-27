import numpy as np
import pytest
import torch
from src.world_model import m3w_fixed_floor_excess as api
from src.world_model import m3w_fixed_floor_tail as control


def test_combination_loss_not_separate_moment_error():
    y=torch.tensor([[1.,.1,.5,.05]],requires_grad=True)
    p=torch.tensor([[11.,.3,5.5,.15]],requires_grad=True)
    assert float(api.loss(p,y).detach())<1e-14
    assert float((p-y).square().mean().detach())>1
    api.loss(p,y).backward(); assert y.grad is None


def test_both_all_and_easy_excess_are_supervised():
    y=torch.zeros(1,4)
    p=torch.tensor([[1.,.12,1.,.22]],requires_grad=True)
    np.testing.assert_allclose(api.signed(p).detach().numpy(),[[.1,.2]],atol=1e-7)
    np.testing.assert_allclose(float(api.loss(p,y).detach()),.025,atol=1e-7)
    api.loss(p,y).backward(); assert (p.grad!=0).all()
    with pytest.raises(ValueError): api.loss(p,torch.full((1,4),float('nan')))


def test_direct_excess_fit_same_init_draws_as_moment_control_and_resume(tmp_path):
    from scripts.replay_m3w_dimensionless_training import exact
    torch.set_num_threads(4); rng=np.random.default_rng(7)
    x=rng.normal(size=(40,6)).astype(np.float32); sites=np.repeat(['a','b'],20)
    y=np.column_stack((np.ones(40),np.linspace(0,.8,40),np.ones(40)*.5,np.linspace(0,.2,40)))
    y[[0,20]]=np.nan; known=np.isfinite(y).all(1); w=known/known.sum()
    pr=dict(mean=x[known].mean(0),std=x[known].std(0),cost_scale=1.,known=known,weights=w,
            constant=(y[known]*w[known,None]).sum(0),clip=10.,training_sites=['a','b'])
    settings=dict(width=8,steps=8,batch_size=16,learning_rate=.0003,gradient_clip=5.,checkpoint_every=4,heartbeat_every=4)
    kwargs=dict(seed=17,settings=settings,identity={'fixed':True},heartbeat=lambda **kw:None)
    env=np.ones(40)
    api.fit(x,y,sites,env,pr,directory=tmp_path/'full',**kwargs)
    api.fit(x,y,sites,env,pr,directory=tmp_path/'resume',stop_at=3,**kwargs)
    _,r=api.fit(x,y,sites,env,pr,directory=tmp_path/'resume',resume=True,**kwargs)
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=False)
    for k in ('model','optimizer','trace','sampler_rng','torch_rng','draws'): exact(a[k],b[k])
    control.fit(x,y,sites,env,pr,arm='mse',directory=tmp_path/'control',**kwargs)
    c=torch.load(tmp_path/'control/checkpoint.pt',weights_only=False)
    api.assert_matched(a,c); assert r['unknown_rows_sampled']==0
    assert a['objective']=='fixed_floor_signed_excess' and a['step']==8
    with pytest.raises(ValueError): api.fit(x,y,sites,env,pr,directory=tmp_path/'full',**kwargs)
