import numpy as np
import pytest
import torch
from src.world_model.m3w_conditional_cost_head import region_weights, loss_value, fit
from src.world_model.m3w_tempered_cost_head import fit as reference_fit, loss_value as reference_loss
from src.world_model.m3w_bounded_cost_head import predict
from src.world_model.m3w_native_gain_harm import preprocess


def test_weights_preserve_fitting_expectation_and_unknown_exclusion():
    pr = dict(known=np.array([True, True, False]), weights=np.array([.25, .75, 0.]))
    w = region_weights(np.array([True, False, True]), pr)
    assert w[0]/w[1] == 4 and w[2] == 0
    assert np.dot(pr['weights'], w) == pytest.approx(1.)
    with pytest.raises(ValueError):
        region_weights(np.array([True, False, True]), pr, multiplier=0)


def test_no_label_or_weight_gradients_and_constant_weight_equivalence():
    p = torch.tensor([[1., 0.], [10., 0.], [0., 0.]], requires_grad=True)
    y = torch.zeros_like(p, requires_grad=True)
    d = torch.tensor([1., 10., 0.])
    w = torch.ones(3, requires_grad=True)
    assert torch.allclose(loss_value(p,y,d,w), reference_loss(p,y,d))
    loss_value(p,y,d,w).backward()
    assert y.grad is None and w.grad is None and torch.isfinite(p.grad).all()
    with pytest.raises(ValueError):
        loss_value(p,y,d,torch.zeros(3))


def test_real_resume_and_identical_draws(tmp_path):
    rng = np.random.default_rng(27)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    d = rng.uniform(.1,5,40)
    y = d[:,None]*rng.uniform(0,.3,(40,2)); cv=np.ones(40)
    sites=np.repeat(['a','b'],20); y[0],cv[0]=np.nan,np.nan
    pr=preprocess(x,y,cv,sites,'c'); w=region_weights(np.arange(40)%3==0,pr)
    settings=dict(width=8,learning_rate=.001,steps=12,batch_size=16,gradient_clip=5.,heartbeat_every=4,checkpoint_every=4)
    kw=dict(seed=17,settings=settings,identity={'synthetic':True},heartbeat=lambda **kw:None)
    m,r=fit(x,y,d,sites,pr,w,directory=tmp_path/'full',**kw)
    fit(x,y,d,sites,pr,w,directory=tmp_path/'resume',stop_at=4,**kw)
    m2,r2=fit(x,y,d,sites,pr,w,directory=tmp_path/'resume',resume=True,**kw)
    reference_fit(x,y,d,sites,pr,directory=tmp_path/'reference',**kw)
    np.testing.assert_array_equal(predict(m,x,d,pr,'bounded_native'),predict(m2,x,d,pr,'bounded_native'))
    a=torch.load(tmp_path/'full'/'checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'reference'/'checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(a['draws'],b['draws'])
    assert r['complete'] and r2['complete'] and not r['unknown_rows_sampled']
    bad=w.copy(); bad[1]*=2
    with pytest.raises(AssertionError):
        fit(x,y,d,sites,pr,bad,directory=tmp_path/'resume',resume=True,**kw)
