import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_observation_cost as api


def test_observation_mixture_keeps_unknown_bound_not_zero_error():
    y = torch.ones((3, 7, 5)); d = torch.full((3, 7), 8.)
    q = torch.tensor([0., 1., .25]); out = api.compose(q, y, d)
    torch.testing.assert_close(out[0, :, 1], d[0]); torch.testing.assert_close(out[0, :, 4], d[0])
    assert out[0, :, [0, 2, 3]].count_nonzero() == 0
    torch.testing.assert_close(out[1], y[1]); assert out[2, 0, 1] == 6.25
    with pytest.raises(ValueError): api.compose(q[:, None], y, d)


def toy():
    rng = np.random.default_rng(5); n = 6; dim = 4
    x = rng.normal(size=(n, 7, dim)).astype(np.float32); radius = np.ones((n, 7), np.float32)
    c = dict(x=x, envelope=radius.copy(), scales=np.ones(n, np.float32), allowed=np.ones((n, 7), bool),
        groups=[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])], source_groups=[np.array([0, 1, 2])],
        sites=np.array(['fit']*n), ids=np.arange(n))
    observed = np.array([True, False, True, True, False, True])
    target = np.ones((n, 7, 5), np.float32)*.2; target[~observed] = 0
    bounded = target.copy(); bounded[~observed, :, 1] = 1; bounded[~observed, :, 4] = 1
    pr = dict(mean=np.zeros(dim), std=np.ones(dim), clip=8., rms=np.ones(5, np.float32), bias=np.zeros(5, np.float32))
    return dict(causal=c, preprocess=pr, z=torch.from_numpy(x), truth=torch.from_numpy(target), radius=radius,
        truth_by_arm=dict(bound_aware=torch.from_numpy(bounded)), observed_future=observed,
        defined_by_arm=dict(bound_aware=np.ones(n, bool)))


def test_auxiliary_identifies_support_without_changing_matched_initial_outputs():
    data = toy(); model = api.ObservationCostHead(4, 8, 17, data['preprocess']['bias'], .95)
    seg = torch.tensor([0, 0, 1, 1, 2, 2]); d = torch.from_numpy(data['radius'])
    parts = model(data['z'], d, d, seg, 3); ix = np.arange(6)
    moments = parts['moments']
    assert (moments[..., 4] <= moments[..., 1]+1e-7).all()
    assert (moments[..., 3] <= moments[..., 2]+1e-7).all()
    torch.testing.assert_close(moments[:, :, 2], moments[:, :1, 2].expand(-1, 7))
    a = api.objective(parts, data, ix, seg, 3, 'marginal_support')
    b = api.objective(parts, data, ix, seg, 3, 'identified_support')
    torch.testing.assert_close(a['bound_anchor'], b['bound_anchor'], rtol=0, atol=0)
    torch.testing.assert_close(b['total']-a['total'], b['identification_auxiliary'])
    assert a['observed_supervised_rows'] == 4 and a['unknown_bound_supervised_rows'] == 2
    b['total'].backward(); assert torch.isfinite(model.support.weight.grad).all()


def test_checkpoint_resume_and_inference_target_rejection(tmp_path):
    data = toy(); settings = dict(width=8, learning_rate=.001, query_batch_size=2,
        minimum_updates=4, largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    kw = dict(arm='identified_support', seed=17, settings=settings, observation_prior=.95, identity=dict(test=1))
    resumed = tmp_path/'resume.pt.gz'; direct = tmp_path/'direct.pt.gz'
    api.fit(data, path=resumed, stop_at=2, **kw)
    a = api.fit(data, path=resumed, resume=True, **kw); b = api.fit(data, path=direct, **kw)
    for key in a:
        if key != 'seconds': api.base.exact(a[key], b[key])
    p = api.predict(a, data['causal'], data['radius'])
    assert p['costs'].shape == (6, 7, 5) and p['observation_probability'].shape == (6,)
    illegal = copy.deepcopy(data['causal']); illegal['valid_future'] = np.ones((6, 12), bool)
    with pytest.raises(ValueError): api.predict(a, illegal, data['radius'])
    with pytest.raises(AssertionError): api.fit(data, path=resumed, resume=True, **dict(kw, observation_prior=.9))
