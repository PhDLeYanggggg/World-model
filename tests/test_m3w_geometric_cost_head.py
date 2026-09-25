import numpy as np
import pytest
import torch

from src.world_model.m3w_geometric_cost_head import (
    rollout_envelope, EnvelopeCostHead, initialize_head, fit, predict,
)
from src.world_model.m3w_native_gain_harm import preprocess


def test_envelope_covers_arbitrary_partial_future_support():
    b = np.zeros((2, 12, 2))
    p = b.copy(); p[:, -1, 0] = 12
    bound = rollout_envelope(b, p)
    np.testing.assert_array_equal(bound, [12, 12])
    target = np.zeros_like(b);target[1,-1,0] = 20
    for mask in (np.ones(12, bool), np.arange(12)==11):
        eb = np.linalg.norm(b[:,mask]-target[:,mask],axis=-1).mean(1)
        ep = np.linalg.norm(p[:,mask]-target[:,mask],axis=-1).mean(1)
        assert np.all(np.abs(eb-ep) <= bound)
    assert np.linalg.norm(p-b,axis=-1).mean(1)[0] < bound[0]


def test_envelope_is_translation_invariant_and_future_free():
    rng = np.random.default_rng(9)
    b,p = rng.normal(size=(2,5,12,2))
    np.testing.assert_allclose(rollout_envelope(b,p),rollout_envelope(b+43,p+43))
    np.testing.assert_array_equal(rollout_envelope(b,b),np.zeros(5))
    with pytest.raises(ValueError):
        rollout_envelope(b,p[:,:11])


@pytest.mark.parametrize('task',['utility','all','easy'])
def test_bounded_outputs_equality_and_gradient(task):
    model = EnvelopeCostHead(3, 8, task)
    x = torch.randn(7,3);d=torch.arange(7,dtype=torch.float32)
    y = model(x,d)
    assert y.shape==(7,2) and torch.isfinite(y).all() and (y>=0).all()
    assert torch.all(y[:,1] <= d)
    if task=='utility':
        assert torch.all(y.sum(1) <= d+1e-6)
        assert torch.equal(y[0],torch.zeros(2))
    else:
        assert y[0,0]>0 and y[0,1]==0
    loss=(y-torch.ones_like(y)).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_only_initial_constants_and_no_mutation():
    pr=dict(constant=np.array([2.,1.]),cost_scale=10.,mean=np.zeros(3))
    model=initialize_head(8,pr,17,'utility',5.)
    result=model(torch.randn(4,3),torch.full((4,),.5)).detach().numpy()*10
    np.testing.assert_allclose(result,np.tile([2.,1.],(4,1)),atol=1e-6)
    risk=initialize_head(8,pr,17,'all',5.)
    out=risk(torch.randn(4,3),torch.zeros(4)).detach().numpy()*10
    np.testing.assert_allclose(out[:,0],2.,atol=1e-6)
    np.testing.assert_array_equal(out[:,1],np.zeros(4))
    np.testing.assert_array_equal(pr['constant'],[2.,1.])


def test_reject_nonfinite_or_negative_bound():
    model=EnvelopeCostHead(3,8,'utility')
    with pytest.raises(ValueError):
        model(torch.zeros(2,3),torch.tensor([1.,-1.]))
    with pytest.raises(ValueError):
        rollout_envelope(np.zeros((1,12,2)),np.full((1,12,2),np.nan))


def test_resume_matches_uninterrupted_and_preserves_training_sampler(tmp_path):
    x=np.arange(48,dtype=float).reshape(16,3)/48
    y=np.tile([.2,.1],(16,1));cv=np.ones(16);sites=np.array(['a','b']*8)
    y[-1]=np.nan;cv[-1]=np.nan
    pr=preprocess(x,y,cv,sites,'held')
    settings=dict(width=8,steps=8,batch_size=4,learning_rate=.001,gradient_clip=5.,
        checkpoint_every=2,heartbeat_every=2)
    kwargs=dict(seed=17,task='utility',settings=settings,identity={'test':'resume'},heartbeat=lambda **_:None)
    model,_=fit(x,y,sites,np.ones(16),pr,directory=tmp_path/'full',**kwargs)
    fit(x,y,sites,np.ones(16),pr,directory=tmp_path/'resumed',stop_at=4,**kwargs)
    restored,report=fit(x,y,sites,np.ones(16),pr,directory=tmp_path/'resumed',resume=True,**kwargs)
    np.testing.assert_array_equal(predict(model,x,np.ones(16),pr),predict(restored,x,np.ones(16),pr))
    assert report['new_updates']==4 and report['unknown_rows_sampled']==0
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(a['draws'],b['draws'])
