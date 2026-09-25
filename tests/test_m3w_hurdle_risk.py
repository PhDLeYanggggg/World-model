import numpy as np
import pytest
import torch

from src.world_model.m3w_hurdle_risk import HurdleRiskHead, factor_targets, objective, fit, predict
from src.world_model.m3w_native_gain_harm import preprocess


def test_joint_event_target_preserves_zeros_unknown_and_reference_mass():
    y=np.array([[2.,0.],[.5,.2],[0.,0.],[np.nan,np.nan]])
    d=np.array([0.,1.,4.,1.])
    out=factor_targets(y,d)
    np.testing.assert_array_equal(out['known'],[True,True,True,False])
    np.testing.assert_array_equal(out['positive'],[False,True,False,False])
    np.testing.assert_allclose(out['fraction'],[0.,.2,0.,0.])
    with pytest.raises(ValueError):factor_targets(np.array([[1.,.1]]),np.array([0.]))
    with pytest.raises(ValueError):factor_targets(np.array([[1.,2.]]),np.array([1.]))


def test_past_only_signature_and_product_moments():
    import inspect
    assert list(inspect.signature(HurdleRiskHead.forward).parameters)==['self','x','envelope']
    m=HurdleRiskHead(3,8)
    z=torch.randn(5,3);d=torch.arange(5,dtype=torch.float32)
    parts=m.components(z,d);p=m(z,d)
    assert p.shape==(5,2) and torch.isfinite(p).all()
    assert p[0,0]>0 and p[0,1]==0
    torch.testing.assert_close(p[:,1],d*parts['probability']*parts['severity'])
    assert torch.all(p[:,1]<=d)


def test_auxiliary_loss_identifies_factors_not_only_product():
    x=torch.zeros(2,3);d=torch.ones(2);target=torch.tensor([[1.,.2],[1.,0.]])
    a,b=HurdleRiskHead(3,8),HurdleRiskHead(3,8)
    with torch.no_grad():
        for m,p,s in [(a,.5,.4),(b,.8,.25)]:
            m.network[-1].weight.zero_()
            m.network[-1].bias.copy_(torch.tensor([.54132485,np.log(p/(1-p)),np.log(s/(1-s))]))
    la,_=objective(a.components(x,d),target,d,'product_mse')
    lb,_=objective(b.components(x,d),target,d,'product_mse')
    torch.testing.assert_close(la,lb)
    ha,_=objective(a.components(x,d),target,d,'hurdle')
    hb,_=objective(b.components(x,d),target,d,'hurdle')
    assert abs(float((ha-hb).detach()))>.01


@pytest.mark.parametrize('arm',['product_mse','hurdle'])
def test_no_positive_batch_finite_gradient(arm):
    m=HurdleRiskHead(3,8);d=torch.ones(4);target=torch.zeros(4,2)
    loss,parts=objective(m.components(torch.zeros(4,3),d),target,d,arm)
    loss.backward()
    assert parts['conditional_mse']==0 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_nonfinite_bounds_rejected():
    m=HurdleRiskHead(3,8)
    with pytest.raises(ValueError):m(torch.zeros(2,3),torch.tensor([1.,-1.]))
    with pytest.raises(ValueError):factor_targets(np.array([[np.nan,0.]]),np.ones(1))


def test_resume_exact_draws_and_unknown_mask(tmp_path):
    torch.set_num_threads(1)
    x=np.arange(48,dtype=float).reshape(16,3)/48
    y=np.tile([.2,.1],(16,1));y[::3,1]=0;y[-1]=np.nan
    cv=np.ones(16);cv[-1]=np.nan;sites=np.array(['a','b']*8)
    pr=preprocess(x,y,cv,sites,'held')
    settings=dict(width=8,steps=8,batch_size=4,learning_rate=.001,gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    kw=dict(seed=17,arm='hurdle',settings=settings,identity={'test':'resume'},heartbeat=lambda **_:None)
    a,_=fit(x,y,sites,np.ones(16),pr,directory=tmp_path/'full',**kw)
    fit(x,y,sites,np.ones(16),pr,directory=tmp_path/'part',stop_at=4,**kw)
    b,r=fit(x,y,sites,np.ones(16),pr,directory=tmp_path/'part',resume=True,**kw)
    np.testing.assert_array_equal(predict(a,x,np.ones(16),pr),predict(b,x,np.ones(16),pr))
    assert r['unknown_rows_sampled']==0 and r['new_updates']==4
    sa=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    sb=torch.load(tmp_path/'part/checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(sa['draws'],sb['draws'])
