import numpy as np
import pytest
import torch

from src.world_model import m3w_ranked_hurdle as old
from src.world_model import m3w_supported_rank_pairs as new
from src.world_model.m3w_hurdle_risk import predict
from src.world_model.m3w_native_gain_harm import preprocess


def test_supported_first_recovers_pairs_separated_by_undefined_labels():
    y = torch.tensor([[1., 0.], [0., 0.], [1., 1.], [0., 0.]])
    p = torch.tensor([[1., .1], [2., 1.], [1., .8], [3., 1.]], requires_grad=True)
    sites = np.array(['a'] * 4)
    _, before = old.ranking_loss(p, y, sites, epsilon=1e-6)
    loss, after = new.ranking_loss(p, y, sites, epsilon=1e-6)
    assert before['rank_pairs'] == 0 and after['rank_pairs'] == 2
    assert after['rank_supported_rows'] == 2
    loss.backward()
    assert torch.isfinite(p.grad).all()
    torch.testing.assert_close(p.grad[[1, 3]], torch.zeros(2, 2))


def test_all_supported_loss_and_gradients_are_exactly_unchanged():
    sites = np.array(['a', 'b', 'a', 'b', 'a', 'b'])
    y = torch.tensor([[1., 0.], [0., 1.], [2., .5], [1., .3], [1., .8], [3., 0.]])
    p = torch.tensor([[1., .1], [.1, 1.], [1., .6], [2., .1], [2., .5], [1., .3]], requires_grad=True)
    q = p.detach().clone().requires_grad_(True)
    a, ai = old.ranking_loss(p, y, sites, epsilon=1e-6)
    b, bi = new.ranking_loss(q, y, sites, epsilon=1e-6)
    a.backward(); b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert all(ai[k] == bi[k] for k in ai)


def test_locality_singletons_and_ties_are_not_forced_to_rank():
    y = torch.tensor([[1., .5], [0., 0.], [2., 1.], [0., 1.]])
    p = torch.ones(4, 2, requires_grad=True)
    loss, info = new.ranking_loss(p, y, np.array(['a', 'b', 'a', 'b']), epsilon=1e-6)
    assert info['rank_pairs'] == 0
    loss.backward()
    torch.testing.assert_close(p.grad, torch.zeros_like(p))


def test_zero_reference_positive_harm_is_supported_but_empty_mass_is_not():
    y = torch.tensor([[0., 1.], [0., 0.], [1., 0.]])
    loss, info = new.ranking_loss(torch.ones_like(y), y, np.array(['a'] * 3), epsilon=1e-6)
    assert torch.isfinite(loss) and info['rank_pairs'] == 2
    assert info['rank_supported_rows'] == 2


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_bad_label_cannot_hide_behind_support_filter(bad):
    y = torch.tensor([[1., 0.], [bad, 0.], [1., 1.]])
    with pytest.raises(ValueError):
        new.ranking_loss(torch.ones_like(y), y, np.array(['a'] * 3), epsilon=1e-6)


def test_no_supported_rows_is_finite_zero_loss():
    p = torch.ones(5, 2, requires_grad=True)
    loss, info = new.ranking_loss(p, torch.zeros_like(p), np.array(['a'] * 5), epsilon=1e-6)
    assert loss.detach().item() == 0 and info['rank_pairs'] == 0 and info['rank_supported_rows'] == 0
    loss.backward()
    torch.testing.assert_close(p.grad, torch.zeros_like(p))


def fixture(*, sparse=True):
    x = np.arange(96, dtype=float).reshape(32, 3) / 96
    y = np.tile([.3, .15], (32, 1)); y[::3, 1] = 0
    if sparse:
        y[1::4] = 0
    y[-1] = np.nan
    cv = np.ones(32); cv[-1] = np.nan
    sites = np.array(['a', 'b'] * 16)
    pr = preprocess(x, y, cv, sites, 'held')
    settings = dict(width=8, steps=8, batch_size=16, learning_rate=.001,
                    gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    return x, y, sites, np.ones(32), pr, settings


@pytest.mark.parametrize('sparse,weight', [(False, 1.), (True, 0.)])
def test_matching_fit_and_draws_when_pairing_has_no_effect(tmp_path, sparse, weight):
    torch.set_num_threads(1)
    x, y, sites, d, pr, settings = fixture(sparse=sparse)
    kw = dict(seed=17, settings=settings, identity={'test': 'equivalence'}, heartbeat=lambda **_: None,
              rank_weight=weight, epsilon=1e-6)
    a, _ = old.fit(x, y, sites, d, pr, directory=tmp_path/'old', **kw)
    b, report = new.fit(x, y, sites, d, pr, directory=tmp_path/'new', **kw)
    np.testing.assert_array_equal(predict(a, x, d, pr), predict(b, x, d, pr))
    one = torch.load(tmp_path/'old/checkpoint.pt', weights_only=False)
    two = torch.load(tmp_path/'new/checkpoint.pt', weights_only=False)
    np.testing.assert_array_equal(one['draws'], two['draws'])
    assert report['unknown_rows_sampled'] == 0
    assert len({r['batch_sha256'] for r in report['fixed_trace']}) == 1
    assert report['fixed_trace'][0]['step'] == 0 and report['fixed_trace'][-1]['step'] == 8


def test_resume_keeps_fixed_batch_diagnostics_and_parameters_exact(tmp_path):
    torch.set_num_threads(1)
    x, y, sites, d, pr, settings = fixture()
    kw = dict(seed=29, settings=settings, identity={'test': 'resume'}, heartbeat=lambda **_: None,
              rank_weight=1., epsilon=1e-6)
    a, full = new.fit(x, y, sites, d, pr, directory=tmp_path/'full', **kw)
    new.fit(x, y, sites, d, pr, directory=tmp_path/'part', stop_at=4, **kw)
    b, resumed = new.fit(x, y, sites, d, pr, directory=tmp_path/'part', resume=True, **kw)
    np.testing.assert_array_equal(predict(a, x, d, pr), predict(b, x, d, pr))
    assert full['fixed_trace'] == resumed['fixed_trace']
    assert full['rank_pairs_seen'] == resumed['rank_pairs_seen']
    assert resumed['new_updates'] == 4
    with pytest.raises(ValueError):
        new.fit(x, y, sites, d, pr, directory=tmp_path/'part', resume=True, **dict(kw, epsilon=.01))
