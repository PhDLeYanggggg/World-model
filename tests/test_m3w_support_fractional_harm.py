import numpy as np
import pytest
import torch
from src.world_model import m3w_support_fractional_harm as method
from src.world_model import m3w_selected_risk_learning as base
from src.world_model.m3w_native_gain_harm import preprocess


def fixture():
    rng=np.random.default_rng(10); x=rng.normal(size=(90,5)).astype(np.float32)
    sites=np.repeat(['a','b','c'],30); env=np.maximum(x[:,0],0).astype(float)
    cv=np.exp(x[:,1]); r=np.exp(x[:,2]); h=env*np.clip(x[:,3]/5+.4,0,1)
    y=base.event_targets(cv,r,r+h,.8); y[1]=np.nan; cv[1]=np.nan
    pr=preprocess(x,y[:,:2],cv,sites,'held')
    masks=np.column_stack((np.ones(90,bool),env>0,env>0))
    return x,y,sites,env,masks,pr


def test_fractional_target_support_and_unknown():
    x,y,s,e,m,p=fixture(); q,use,count=method.fractional_targets(y,e)
    assert not use[1] and np.isfinite(q).all() and count==0
    np.testing.assert_allclose(q[use],y[use][:,[1,3]]/e[use,None])
    assert (q[~use]==0).all()


def test_invalid_zero_envelope_harm_rejected():
    y=np.array([[1.,.1,1.,.1]])
    with pytest.raises(ValueError): method.fractional_targets(y,np.zeros(1))


def test_support_auxiliary_ignores_structural_zero_and_is_site_balanced():
    pred=torch.tensor([[1.,0.,1.,0.],[1.,.2,1.,.1],[1.,.4,1.,.2]],requires_grad=True)
    env=torch.tensor([0.,1.,1.]); q=torch.tensor([[0.,0.],[.3,.1],[.2,.1]])
    loss,n,k=method.auxiliary(pred,env,q,np.array(['a','a','b']))
    expected=torch.nn.functional.binary_cross_entropy(pred[1:][:,[1,3]],q[1:])
    assert torch.allclose(loss,expected,atol=1e-7,rtol=1e-7) and (n,k)==(2,2)
    ids=torch.tensor([0,1,1,2])
    duplicate,_,_=method.auxiliary(pred[ids],env[ids],q[ids],np.array(['a','a','a','b']))
    assert torch.allclose(loss,duplicate,atol=1e-7,rtol=1e-7)
    loss.backward(); assert torch.equal(pred.grad[0],torch.zeros(4))


def test_fractional_crossentropy_elicits_soft_mean_not_event_prevalence():
    target=torch.tensor([[.2,.1],[.4,.3]])
    p=torch.tensor([[1.,.3,1.,.2],[1.,.3,1.,.2]],requires_grad=True)
    loss,_,_=method.auxiliary(p,torch.ones(2),target,np.array(['a','a']))
    loss.backward(); assert torch.allclose(p.grad.sum(0),torch.zeros(4),atol=1e-6)


def test_zero_coefficient_matches_original_and_resume(tmp_path):
    torch.set_num_threads(1)
    args=fixture(); settings=dict(width=8,steps=12,batch_size=16,learning_rate=.0003,
        gradient_clip=5.,checkpoint_every=4,heartbeat_every=4)
    kw=dict(seed=17,settings=settings,identity={'test':True},heartbeat=lambda **v:None)
    original,a=base.fit(*args,arm='mean',directory=tmp_path/'original',**kw)
    matched,b=method.fit(*args,coefficient=0,directory=tmp_path/'matched',**kw)
    for k,v in original.state_dict().items(): assert torch.equal(v,matched.state_dict()[k])
    direct,_=method.fit(*args,directory=tmp_path/'direct',**kw)
    method.fit(*args,directory=tmp_path/'resume',stop_at=5,**kw)
    resumed,_=method.fit(*args,directory=tmp_path/'resume',resume=True,**kw)
    for k,v in direct.state_dict().items(): assert torch.equal(v,resumed.state_dict()[k])
    s=[torch.load(tmp_path/p/'checkpoint.pt',weights_only=False) for p in ('original','matched','direct','resume')]
    for state in s[1:]:
        np.testing.assert_array_equal(s[0]['draws'],state['draws'])
        assert torch.equal(s[0]['sampler_rng'],state['sampler_rng'])
    assert a['unknown_rows_sampled']==b['unknown_rows_sampled']==0
