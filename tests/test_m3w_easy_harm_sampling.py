import numpy as np
import pytest
import torch
from src.world_model.m3w_easy_harm_sampling import probabilities, per_row_loss, draw, fit


def fixture():
    rng = np.random.default_rng(18); n = 40
    x = rng.normal(size=(n, 6)); env = np.full(n, 2.)
    y = np.column_stack((np.ones(n), np.full(n, .2), np.full(n, .5), np.zeros(n)))
    y[[2, 25], 3] = [.1, .2]; y[-1] = np.nan
    sites = np.repeat(['a', 'b'], 20); p = np.r_[np.full(20, .5/20), np.full(19, .5/19), 0.]
    pr = dict(mean=np.zeros(6), std=np.ones(6), known=p > 0, weights=p, cost_scale=1.)
    return x, y, sites, env, np.ones((n, 3), bool), pr


def test_mixture_preserves_sites_and_expected_loss_gradient():
    x,y,sites,env,masks,pr = fixture(); p = pr['weights']; q,w = probabilities(y,sites,p)
    np.testing.assert_allclose(q*w,p,atol=1e-15)
    assert w.max() <= 2. and q[-1] == 0 and q[2] > p[2]
    for site in ('a','b'): assert q[sites == site].sum() == pytest.approx(.5)
    pred = torch.tensor(np.ones((len(x),4)), dtype=torch.float64, requires_grad=True)
    target = torch.tensor(np.nan_to_num(y)); scales = torch.ones(4, dtype=torch.float64)
    loss = per_row_loss(pred,target,scales)
    a = (torch.tensor(p)*loss).sum(); b = (torch.tensor(q)*torch.tensor(w)*loss).sum()
    ga = torch.autograd.grad(a,pred,retain_graph=True)[0]; gb = torch.autograd.grad(b,pred)[0]
    torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(ga,gb,rtol=1e-14,atol=1e-14)


def test_no_harm_site_uses_base_distribution_and_unknowns_never_draw():
    _,y,s,_,_,pr = fixture(); y[:20,3] = 0; q,w = probabilities(y,s,pr['weights'])
    np.testing.assert_array_equal(q[:20],pr['weights'][:20])
    cdf = torch.tensor(q.cumsum()); cdf[-1] = 1.
    ids = draw(cdf,10000,torch.Generator().manual_seed(4)); assert (ids < 39).all()


def test_unknown_as_zero_sampling_weight_rejected():
    _,y,s,_,_,pr = fixture(); p = np.full(len(y),1/len(y))
    with pytest.raises(ValueError): probabilities(y,s,p)


def test_corrected_training_resume_is_exact(tmp_path):
    args = fixture(); settings = dict(width=5,steps=6,batch_size=8,learning_rate=.001,
        gradient_clip=5.,checkpoint_every=3,heartbeat_every=3)
    common = dict(seed=17,settings=settings,identity={'test':True},heartbeat=lambda **kw:None)
    full,_ = fit(*args, directory=tmp_path/'full',**common)
    fit(*args,directory=tmp_path/'resume',stop_at=3,**common)
    resumed,metrics = fit(*args,directory=tmp_path/'resume',resume=True,**common)
    for k,v in full.state_dict().items(): assert torch.equal(v,resumed.state_dict()[k])
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(a['draws'],b['draws']); assert metrics['unknown_rows_sampled'] == 0
