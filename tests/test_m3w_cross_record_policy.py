import copy
import json

import pytest
import torch

from src.world_model import m3w_cross_record_policy as api
from tests.test_m3w_direct_policy import fixture, settings


def data():
    return api.base.prepare(*fixture())


def config():
    return {**settings(), 'chunk_size': 2, 'inner_learning_rate': .01,
            'inner_gradient_clip': 5., 'meta_weight': .5}


def test_episode_partition_is_input_only_exhaustive_and_record_disjoint():
    keys = [json.dumps([s, str(r)]) for s, n in [('a', 5), ('b', 1), ('c', 1)] for r in range(n)]
    a = api.partition(keys, 17, 0)
    assert a.dtype == torch.bool and a.any() and (~a).any()
    assert torch.equal(a, api.partition(keys, 17, 0))
    assert 0 < int(a[:5].sum()) < 5
    assert a[5] != a[6]
    assert any(not torch.equal(a, api.partition(keys, 17, i)) for i in range(1, 8))
    with pytest.raises(ValueError):
        api.partition(keys[:1], 17, 0)


@pytest.mark.parametrize('arm', api.ARMS)
def test_zero_inner_step_recovers_exact_recording_gradient(arm):
    d = data(); loc = api.whole.layout(d); dual = torch.full((len(d['groups']), 2), .7)
    m = api.base.Policy(d['z'].shape[1], 8, 17)
    stats = api.whole.full_pass(m, d, loc, dual, .5, 2, backward=True)
    expected = torch.cat([p.grad.flatten() for p in m.parameters()]).clone()
    m.zero_grad(set_to_none=True)
    loss, info = api.meta_loss(m, d, loc, dual, .5, api.partition(d['record_keys'], 17, 0),
                               arm, {**config(), 'inner_learning_rate': 0.})
    loss.backward()
    actual = torch.cat([p.grad.flatten() for p in m.parameters()])
    torch.testing.assert_close(expected, actual, atol=2e-7, rtol=3e-5)
    assert float(loss.detach()) == pytest.approx(stats['loss'], abs=1e-6)
    torch.testing.assert_close(info['risk'], stats['risk'])


@pytest.mark.parametrize('arm', api.ARMS)
def test_second_order_gradient_matches_dense_reference_without_mutating_model(arm):
    d = data(); loc = api.whole.layout(d); dual = torch.full((len(d['groups']), 2), .7)
    m = api.base.Policy(d['z'].shape[1], 8, 17)
    with torch.no_grad():
        m.net[-1].weight.fill_(.03)
    initial = copy.deepcopy(m.state_dict()); side = api.partition(d['record_keys'], 17, 1)
    cfg = {**config(), 'inner_learning_rate': .2, 'inner_gradient_clip': 1000.}
    loss, _ = api.meta_loss(m, d, loc, dual, .5, side, arm, cfg)
    actual = torch.autograd.grad(loss, tuple(m.parameters()))
    params = dict(m.named_parameters()); ix, rec = loc['index'], loc['record']
    coefficient = -d['gain'][ix]+(dual[rec]*d['risk'][ix]/d['record_scales'][rec]).sum(1)
    weight = loc['weight']; wa = d['weights'][side].sum(); wb = d['weights'][~side].sum()

    def objective(p, selected):
        mask = selected[rec]
        z = torch.func.functional_call(m, p, (d['z'][ix], d['anchor'][ix]), strict=True)
        a = api.base.decision(z, True, .5)
        return (a[mask]*coefficient[mask]*weight[mask]).sum()/d['weights'][selected].sum()

    la, lb = objective(params, side), objective(params, ~side)
    ga = torch.autograd.grad(la, tuple(params.values()), create_graph=True)
    gb = torch.autograd.grad(lb, tuple(params.values()), create_graph=True)
    pa = {k:p-.2*g for (k,p),g in zip(params.items(),ga)}
    pb = {k:p-.2*g for (k,p),g in zip(params.items(),gb)}
    if arm == 'cross_record_meta':
        query = wb*objective(pa, ~side)+wa*objective(pb, side)
    else:
        query = wa*objective(pa, side)+wb*objective(pb, ~side)
    expected = torch.autograd.grad(.5*(wa*la+wb*lb)+.5*query, tuple(params.values()))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=3e-5)
    api.base.core.exact(initial, m.state_dict())


@pytest.mark.parametrize('arm', api.ARMS)
def test_exact_resume_and_real_updates(tmp_path, arm):
    d = data(); kw = dict(arm=arm, settings=config(), seed=17, identity={'test': True})
    full = api.fit(d, path=tmp_path/'full.pt.gz', **kw)
    api.fit(d, path=tmp_path/'split.pt.gz', stop_at=2, **kw)
    split = api.fit(d, path=tmp_path/'split.pt.gz', resume=True, **kw)
    for key in full:
        if key != 'seconds': api.base.core.exact(full[key], split[key])
    assert any(not torch.equal(full['model'][k], v) for k,v in full['initial_model'].items())
    assert full['meta_forward_rows'] == 2*6*sum(d['sizes'])
    assert full['optimizer_steps'] == 6 and not full['test_time_adaptation']
    assert full['second_order_gradient'] and not full['held_used_for_training']


def test_differentiable_inner_clip_bounds_parameter_step():
    m = torch.nn.Linear(2, 1)
    ps = dict(m.named_parameters())
    loss = sum(p.square().sum()*100 for p in ps.values())
    adapted, norm, scale = api.adapt(ps, loss, .1, .2)
    displacement = torch.sqrt(sum((ps[k]-p).square().sum() for k,p in adapted.items()))
    assert float(displacement.detach()) <= .020001 and scale < 1 and norm > .2
    gradient = torch.autograd.grad(sum(p.square().sum() for p in adapted.values()), tuple(ps.values()))
    assert all(torch.isfinite(g).all() for g in gradient)
