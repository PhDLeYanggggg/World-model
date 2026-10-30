import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_selected_query_critic as api


def fixture():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 7, 4)).astype(np.float32)
    p = np.broadcast_to([.2, .01, .5, .2, .001], (8, 7, 5)).copy()
    env = np.ones((8, 7))
    initial = np.array([0, 1, -1, -1, 2, -1, 1, 0])
    c = api.causal_inputs(x, p, env, np.ones(8), np.ones((8, 7), bool), initial,
        np.repeat(['a', 'b'], 4), np.repeat(['r1', 'r2'], 4), np.tile([0, 0, 1, 1], 2), np.arange(8))
    y = np.broadcast_to([.1, .04, .5, .2, .02], (8, 7, 5)).copy()
    y[1] = np.nan
    return c, y


def settings():
    return dict(width=8, steps=6, query_batch_size=4, learning_rate=.001,
        gradient_clip=5., heartbeat_every=2, checkpoint_every=2)


def test_identity_and_reference_protection():
    c, _ = fixture()
    model = api.initialize(4, 8, 17)
    args = api.batch(c, np.arange(8), np.repeat(np.arange(4), 2), 4)
    for arm in api.ARMS:
        p = model(*args, arm=arm)
        torch.testing.assert_close(p, args[1], rtol=0, atol=0)
        p[:, :, [1, 4]].sum().backward()
        assert model.output.weight.grad.abs().sum() > 0
        model.zero_grad()


def test_context_uses_actual_selected_set_and_current_query_only():
    h = torch.arange(8*7*3, dtype=torch.float32).reshape(8, 7, 3)
    c, _ = fixture(); segments = torch.repeat_interleave(torch.arange(4), 2)
    context, counts = api.selected_context(h, torch.from_numpy(c['initial']), segments, 4)
    torch.testing.assert_close(context[0], h[[0, 1], [0, 1]].mean(0))
    torch.testing.assert_close(context[2], torch.zeros(3))
    assert counts.tolist() == [2, 2, 0, 0, 1, 1, 2, 2]
    changed = h.clone(); changed[3:6] += 1000
    other, _ = api.selected_context(changed, torch.from_numpy(c['initial']), segments, 4)
    torch.testing.assert_close(other[:4], context[:4])
    order = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
    perm, _ = api.selected_context(h[order], torch.from_numpy(c['initial'])[order], segments[order], 4)
    torch.testing.assert_close(perm, context[order])


def test_unknown_future_rows_remain_in_context_but_not_loss():
    c, y = fixture(); d = api.training_data(c, y)
    assert len(c['ids']) == 8 and d['known'].sum() == 7
    groups = c['groups']; ix = np.concatenate(groups)
    seg = np.repeat(np.arange(len(groups)), [len(v) for v in groups])
    args = api.batch(c, ix, seg, len(groups))
    model = api.initialize(4, 8, 17)
    p = model(*args, arm='selected_set')
    terms = api.losses(p, d, ix, torch.from_numpy(seg), len(groups))
    assert terms['unknown_supervised_rows'] == 0
    assert terms['complete_selected_queries'] == 2
    assert torch.isfinite(terms['total'])
    y2 = y.copy(); y2[1] = np.nan
    torch.testing.assert_close(model(*args, arm='selected_set'), p)
    with pytest.raises(TypeError):
        api.causal_inputs(**dict(future_endpoint=np.zeros(8)))


@pytest.mark.parametrize('arm', api.ARMS)
def test_real_training_resume_exact_and_reference_immutable(tmp_path, arm):
    c, y = fixture(); d = api.training_data(c, y)
    def fit(path, **kw):
        return api.fit(d, arm=arm, seed=17, settings=settings(), path=path,
            identity=dict(test=True), **kw)
    direct = fit(tmp_path/'direct.pt.gz')
    fit(tmp_path/'resume.pt.gz', stop_at=2)
    resumed = fit(tmp_path/'resume.pt.gz', resume=True)
    for k in direct:
        if k != 'seconds':
            api.exact(direct[k], resumed[k])
    assert direct['unknown_rows_drawn'] > 0 and direct['unknown_rows_with_loss'] == 0
    assert direct['step'] == 6
    assert any(not torch.equal(v, direct['initial_model'][k]) for k, v in direct['model'].items())
    p = api.predict(direct, c)
    np.testing.assert_array_equal(p[:, :, [0, 2, 3]], c['prediction'][:, :, [0, 2, 3]])
    assert np.isfinite(p).all() and (p >= 0).all()
    assert (p[:, :, 4] <= p[:, :, 1]+1e-12).all()
    assert (p[:, :, :2].sum(-1) <= c['envelope']+1e-12).all()
    bad = copy.deepcopy(c); bad['prediction'][0, 0, 0] += .01
    changed = api.training_data(bad, y)
    with pytest.raises((AssertionError, ValueError)):
        api.fit(changed, arm=arm, seed=17, settings=settings(), path=tmp_path/'resume.pt.gz',
            identity=dict(test=True), resume=True)


def test_two_arms_have_identical_initialization_draws_and_risk_normalizers(tmp_path):
    c, y = fixture(); d = api.training_data(c, y)
    states = [api.fit(d, arm=a, seed=17, settings=settings(), path=tmp_path/(a+'.pt.gz'),
        identity=dict(test=True)) for a in api.ARMS]
    api.assert_matched(*states)
    assert states[0]['parameters'] == states[1]['parameters']
    assert len(c['groups']) == 4
    assert set(c['keys']) == {('a', 'r1', 0), ('a', 'r1', 1), ('b', 'r2', 0), ('b', 'r2', 1)}


def test_risk_encoder_is_trainable_and_inference_rejects_labels(tmp_path):
    c, y = fixture(); model = api.initialize(4, 8, 17)
    with torch.no_grad():
        model.output.weight.fill_(.01)
    args = api.batch(c, np.arange(8), np.repeat(np.arange(4), 2), 4)
    model(*args, arm='selected_set')[:, :, [1, 4]].sum().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    state = api.fit(api.training_data(c, y), arm='selected_set', seed=17,
        settings=settings(), path=tmp_path/'head.pt.gz', identity=dict(test=True))
    with pytest.raises(ValueError):
        api.predict(state, dict(c, targets=y))
