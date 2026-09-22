import numpy as np
import pytest
import torch
from src.world_model.m3w_tempered_cost_head import loss_value, fit
from src.world_model.m3w_bounded_cost_head import fit as reference_fit, predict
from src.world_model.m3w_native_gain_harm import preprocess


def test_fixed_intermediate_loss_and_zero_handling():
    p = torch.tensor([[1., 0.], [10., 0.], [0., 0.]], requires_grad=True)
    d = torch.tensor([1., 10., 0.]); y = torch.zeros_like(p)
    assert float(loss_value(p, y, d).detach()) == pytest.approx(11/6)
    loss_value(p, y, d).backward()
    assert torch.isfinite(p.grad).all() and torch.equal(p.grad[2], torch.zeros(2))
    with pytest.raises(ValueError): loss_value(p, y, -d)


def test_sampler_and_resume_match_reference(tmp_path):
    rng = np.random.default_rng(27); x = rng.normal(size=(40, 5)).astype(np.float32)
    d = rng.uniform(.1, 5, 40); y = d[:, None]*rng.uniform(0, .3, (40, 2)); cv = np.ones(40)
    sites = np.repeat(['a','b'],20); y[0],cv[0]=np.nan,np.nan
    pr = preprocess(x,y,cv,sites,'c')
    settings = dict(width=8,learning_rate=.001,steps=12,batch_size=16,gradient_clip=5.,heartbeat_every=4,checkpoint_every=4)
    kw = dict(seed=17, settings=settings, identity={'synthetic':True}, heartbeat=lambda **kw:None)
    model,r = fit(x,y,d,sites,pr,directory=tmp_path/'full',**kw)
    fit(x,y,d,sites,pr,directory=tmp_path/'resume',stop_at=4,**kw)
    resumed,rr = fit(x,y,d,sites,pr,directory=tmp_path/'resume',resume=True,**kw)
    reference_fit(x,y,d,sites,pr,directory=tmp_path/'reference',arm='bounded_fraction',**kw)
    np.testing.assert_array_equal(predict(model,x,d,pr,'bounded_native'),predict(resumed,x,d,pr,'bounded_native'))
    a = torch.load(tmp_path/'full'/'checkpoint.pt',weights_only=False)
    b = torch.load(tmp_path/'reference'/'checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(a['draws'],b['draws'])
    assert r['complete'] and rr['complete'] and r['unknown_rows_sampled'] == 0
    with pytest.raises(ValueError):
        fit(x,y,d,sites,pr,directory=tmp_path/'resume',resume=True,**(kw|{'identity':{'changed':True}}))
