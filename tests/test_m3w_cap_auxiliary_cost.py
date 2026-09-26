import numpy as np
import pytest
import torch
from src.world_model import m3w_cap_auxiliary_cost as m


def fixture():
    rng = np.random.default_rng(53)
    x = rng.normal(size=(96, 6)); x[::7, 1] = np.nan
    sites = np.repeat(['a', 'b', 'c'], 32)
    env = np.ones(96); env[::11] = 0
    harm = env*rng.uniform(size=96)
    easy = np.arange(96) % 3 == 0
    y = np.column_stack((np.ones(96), harm, easy, harm*easy)).astype(float)
    y[::9] = np.nan
    event = np.where(np.isfinite(y).all(1) & (env > 0), y[:, 3] > .3, np.nan)
    return x, y, event, sites, env


def test_event_shuffle_preserves_each_locality_and_missingness():
    _, _, event, sites, _ = fixture()
    p = m.auxiliary_target(event, sites, 'shuffled_aux', 7)
    np.testing.assert_equal(np.isnan(p), np.isnan(event))
    for site in set(sites):
        np.testing.assert_equal(np.sort(p[sites == site]), np.sort(event[sites == site]))
    assert not np.array_equal(p, event, equal_nan=True)
    np.testing.assert_equal(p, m.auxiliary_target(event, sites, 'shuffled_aux', 7))


def test_event_head_cannot_multiply_or_change_cost_at_inference():
    model = m.CapAuxiliaryCostHead(6, 8)
    x, env = torch.randn(20, 6), torch.linspace(0, 1, 20)
    cost, p = model(x, env)
    with torch.no_grad():
        model.event.weight.fill_(1000); model.event.bias.fill_(-100)
    other, q = model(x, env)
    torch.testing.assert_close(cost, other, rtol=0, atol=0)
    assert not torch.equal(p, q)
    assert torch.all(cost[:, 1] <= cost[:, 0]) and torch.all(cost[:, 0] <= env)
    assert torch.equal(cost[0], torch.zeros(2))


def test_cost_only_gradient_does_not_depend_on_auxiliary_label():
    c = torch.tensor([[.4, .2]], requires_grad=True)
    p = torch.tensor([.4], requires_grad=True)
    args = (torch.zeros_like(c), torch.ones(2))
    a, _ = m.objective(c, p, args[0], torch.zeros(1), args[1], 'cost_only')
    b, _ = m.objective(c, p, args[0], torch.ones(1), args[1], 'cost_only')
    assert a.item() == b.item()
    ga, gb = torch.autograd.grad(a, (c, p), retain_graph=True), torch.autograd.grad(b, (c, p))
    for u, v in zip(ga, gb): torch.testing.assert_close(u, v, rtol=0, atol=0)


def test_preprocess_excludes_unknown_and_outer_and_invalid_costs():
    x, y, e, s, env = fixture()
    pr = m.prepare(x, y, e, s, 'd', env, 2.)
    assert not pr['known'][env == 0].any()
    bad = x.copy(); bad[~pr['known']] = 1e9
    pr2 = m.prepare(bad, y, e, s, 'd', env, 2.)
    for key in ('mean', 'std', 'weights', 'loss_scales'):
        np.testing.assert_array_equal(pr[key], pr2[key])
    with pytest.raises(ValueError): m.prepare(x, y, e, s, 'a', env, 2.)
    illegal = y.copy(); illegal[1, 3] = 3
    with pytest.raises(ValueError): m.prepare(x, illegal, e, s, 'd', env, 2.)


@pytest.mark.parametrize('arm', m.ARMS)
def test_resume_prediction_and_identity(tmp_path, arm):
    torch.set_num_threads(4)
    x, y, e, s, env = fixture()
    settings = dict(width=8, steps=24, batch_size=16, learning_rate=.001,
                    gradient_clip=5, checkpoint_every=8, heartbeat_every=8)
    kw = dict(arm=arm, seed=7, settings=settings, identity={'frozen': True}, heartbeat=lambda **kw: None)
    a, pr, fit = m.fit(x, y, e, s, 'd', env, 2., directory=tmp_path/'full', **kw)
    m.fit(x, y, e, s, 'd', env, 2., directory=tmp_path/'resume', stop_at=8, **kw)
    b, pr2, other = m.fit(x, y, e, s, 'd', env, 2., directory=tmp_path/'resume', resume=True, **kw)
    for u, v in zip(m.predict(a, x, env, pr), m.predict(b, x, env, pr2)):
        np.testing.assert_array_equal(u, v)
    restored, state = m.restore(tmp_path/'resume')
    for u, v in zip(m.predict(restored, x, env, state['preprocess']), m.predict(b, x, env, pr2)):
        np.testing.assert_array_equal(u, v)
    assert fit['complete'] and other['complete'] and fit['unknown_rows_sampled'] == 0
    with pytest.raises(ValueError): m.fit(x, y, e, s, 'd', env, 2., directory=tmp_path/'resume', **kw)


def test_arms_have_identical_initialization_and_sampler(tmp_path):
    x, y, e, s, env = fixture()
    settings = dict(width=8, steps=8, batch_size=16, learning_rate=.001,
                    gradient_clip=5, checkpoint_every=8, heartbeat_every=8)
    states = []
    for arm in m.ARMS:
        m.fit(x, y, e, s, 'd', env, 2., arm=arm, seed=7, settings=settings,
              identity={'fixed': True}, directory=tmp_path/arm, heartbeat=lambda **kw: None)
        states.append(m.restore(tmp_path/arm)[1])
    for state in states[1:]:
        for key in states[0]['initial_model']:
            assert torch.equal(state['initial_model'][key], states[0]['initial_model'][key])
        for key in ('fixed', 'draws'):
            np.testing.assert_array_equal(state[key], states[0][key])
        assert torch.equal(state['sampler_rng'], states[0]['sampler_rng'])
