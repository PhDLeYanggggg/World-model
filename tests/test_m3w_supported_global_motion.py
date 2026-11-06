import numpy as np
import pytest
import torch

from src.world_model import m3w_supported_global_motion as m


def sample():
    r = np.random.default_rng(27); n = 42
    return dict(x=r.normal(size=(n, 120)), context=r.normal(size=(n, 48)),
        moments=np.abs(r.normal(size=(n, 5))), weights=np.ones(n),
        observed=np.tile([True, True, False], 14), envelope=np.ones(n),
        sites=np.repeat(['a', 'b', 'held'], 14), ids=np.arange(n)*7+1)


def test_exact_join_and_rejection():
    values = np.arange(144).reshape(3, 48)
    result = m.lookup(np.array([15, 8, 20]), np.array([2, 1, 9]), values)
    np.testing.assert_array_equal(result, values[[0, 1, 0]])
    with pytest.raises(ValueError): m.lookup(np.array([29]), np.array([2, 1, 9]), values)
    with pytest.raises(ValueError): m.lookup(np.array([14]), np.array([2, 1, 9]), values)


def test_preprocessing_excludes_held_future_and_context():
    d = sample(); a = m.prepare(d, 'held')
    d['context'][28:] = 1e20; d['moments'][28:] = 1e20; d['x'][28:] = 1e20
    b = m.prepare(d, 'held')
    m.base.exact(a['binding'], b['binding'])
    assert a['binding']['train_sources'] == ['a', 'b']


def test_zero_context_branch_starts_at_exact_parent():
    p = m.prepare(sample(), 'held'); x, e, c = p['x'], p['envelope'], p['context']
    torch.manual_seed(17); old = m.parent.Head(m.SETTINGS, p['binding']['parent'])
    torch.manual_seed(17); new = m.Head(m.SETTINGS, p['binding']['parent'])
    for arm in m.ARMS:
        out = new(x, e, c, arm)
        for k, value in old(x, e).items(): torch.testing.assert_close(out[k], value, rtol=0, atol=0)


def test_local_ablation_masks_global_even_after_training_weights():
    p = m.prepare(sample(), 'held'); model = m.Head(m.SETTINGS, p['binding']['parent'])
    for layer in (model.conditional[-1], model.reference[-1], model.observation[-1]):
        torch.nn.init.normal_(layer.weight, std=.1)
    for layer in (model.global_conditional, model.global_reference, model.global_observation):
        torch.nn.init.normal_(layer.weight)
    a = model(p['x'], p['envelope'], p['context'], 'own_local')
    b = model(p['x'], p['envelope'], 100*p['context'], 'own_local')
    for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    assert not torch.equal(a['moments'], model(p['x'], p['envelope'], p['context'], 'own_global')['moments'])


def test_reference_and_support_action_invariant():
    p = m.prepare(sample(), 'held'); model = m.Head(m.SETTINGS, p['binding']['parent'])
    x = p['x'][:6].clone(); x[:, :50] = x[0, :50].clone()
    c = p['context'][:6].clone(); c[:] = c[0].clone()
    for layer in (model.conditional[-1], model.reference[-1], model.observation[-1]):
        torch.nn.init.normal_(layer.weight, std=.1)
    for layer in (model.global_conditional, model.global_reference, model.global_observation):
        torch.nn.init.normal_(layer.weight)
    out = model(x, p['envelope'][:6], c, 'augmented_global')
    assert torch.equal(out['conditional'][:, 2:4], out['conditional'][0, 2:4].expand(6, 2))
    assert torch.equal(out['observation_probability'], out['observation_probability'][0].expand(6))


def test_real_optimizer_resume(tmp_path):
    d = sample(); kwargs = dict(arm='own_global', seed=17)
    m.fit(d, 'held', {'pilot': True}, tmp_path/'resume.pt.gz', stop_at=3, **kwargs)
    a = m.fit(d, 'held', {'pilot': True}, tmp_path/'resume.pt.gz', stop_at=6, resume=True, **kwargs)
    b = m.fit(d, 'held', {'pilot': True}, tmp_path/'direct.pt.gz', stop_at=6, **kwargs)
    m.exact(a, b)
