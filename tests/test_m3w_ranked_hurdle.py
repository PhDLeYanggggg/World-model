import inspect

import numpy as np
import pytest
import torch

from src.world_model import m3w_hurdle_risk as control
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_ranked_hurdle import cyclic_pairs, ranking_loss, fit


def test_pairs_never_cross_localities_and_are_deterministic():
    sites = np.array(['a', 'b', 'a', 'c', 'b', 'a'])
    left, right = cyclic_pairs(sites)
    np.testing.assert_array_equal(sites[left], sites[right])
    assert len(left) == 5 and np.all(left != right)
    np.testing.assert_array_equal(left, cyclic_pairs(sites)[0])
    np.testing.assert_array_equal(right, cyclic_pairs(sites)[1])


def test_ordering_direction_and_finite_gradient():
    target = torch.tensor([[1., 0.], [1., .8]])
    correct = torch.tensor([[1., .01], [1., .6]], requires_grad=True)
    wrong = torch.tensor([[1., .6], [1., .01]], requires_grad=True)
    a, info = ranking_loss(correct, target, np.array(['a', 'a']), epsilon=1e-6)
    b, _ = ranking_loss(wrong, target, np.array(['a', 'a']), epsilon=1e-6)
    assert a < b and info['rank_pairs'] == 2
    b.backward()
    assert torch.isfinite(wrong.grad).all()
    assert wrong.grad[0, 1] > 0 and wrong.grad[1, 1] < 0


def test_undefined_and_equal_risk_labels_do_not_force_rank():
    y = torch.tensor([[0., 0.], [1., .2], [1., .2]])
    p = torch.ones(3, 2, requires_grad=True)
    loss, info = ranking_loss(p, y, np.array(['a', 'a', 'a']), epsilon=1e-6)
    assert float(loss.detach()) == 0 and info['rank_pairs'] == 0
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_zero_reference_positive_harm_is_supported_and_unknown_rejected():
    y = torch.tensor([[0., 1.], [1., 0.]])
    p = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    loss, info = ranking_loss(p, y, np.array(['a', 'a']), epsilon=1e-6)
    assert torch.isfinite(loss) and info['rank_pairs'] == 2
    with pytest.raises(ValueError):
        ranking_loss(p, torch.full_like(y, float('nan')), np.array(['a', 'a']), epsilon=1e-6)
    with pytest.raises(ValueError):
        ranking_loss(p, y, np.array(['a', 'a']), epsilon=0)


def test_rank_weight_tracks_margin_not_hard_class_balance():
    p = torch.tensor([[1., .1], [1., .5]])
    _, far = ranking_loss(p, torch.tensor([[1., 0.], [1., 1.]]), np.array(['a', 'a']), epsilon=1e-6)
    _, near = ranking_loss(p, torch.tensor([[1., .4], [1., .5]]), np.array(['a', 'a']), epsilon=1e-6)
    assert far['rank_weight_sum'] > near['rank_weight_sum'] > 0


def fixture():
    x = np.arange(60, dtype=float).reshape(20, 3) / 60
    y = np.tile([.3, .15], (20, 1)); y[::3, 1] = 0; y[-1] = np.nan
    cv = np.ones(20); cv[-1] = np.nan
    sites = np.array(['a', 'b'] * 10)
    pr = preprocess(x, y, cv, sites, 'held')
    settings = dict(width=8, steps=8, batch_size=8, learning_rate=.001,
                    gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    return x, y, sites, np.ones(20), pr, settings


def test_zero_auxiliary_matches_control_exactly(tmp_path):
    torch.set_num_threads(1)
    x, y, sites, d, pr, settings = fixture()
    kw = dict(seed=17, settings=settings, identity={'test': 'control'}, heartbeat=lambda **_: None)
    a, _ = control.fit(x, y, sites, d, pr, arm='hurdle', directory=tmp_path/'control', **kw)
    b, r = fit(x, y, sites, d, pr, rank_weight=0., epsilon=1e-6, directory=tmp_path/'rank', **kw)
    np.testing.assert_array_equal(control.predict(a, x, d, pr), control.predict(b, x, d, pr))
    assert r['unknown_rows_sampled'] == 0
    assert list(inspect.signature(b.forward).parameters) == ['x', 'envelope']


def test_resume_exact_and_settings_bound(tmp_path):
    torch.set_num_threads(1)
    x, y, sites, d, pr, settings = fixture()
    kw = dict(seed=17, settings=settings, identity={'test': 'resume'}, heartbeat=lambda **_: None,
              rank_weight=1., epsilon=1e-6)
    a, _ = fit(x, y, sites, d, pr, directory=tmp_path/'full', **kw)
    fit(x, y, sites, d, pr, directory=tmp_path/'part', stop_at=4, **kw)
    b, r = fit(x, y, sites, d, pr, directory=tmp_path/'part', resume=True, **kw)
    np.testing.assert_array_equal(control.predict(a, x, d, pr), control.predict(b, x, d, pr))
    assert r['new_updates'] == 4 and r['rank_pairs_seen'] > 0 and r['unknown_rows_sampled'] == 0
    with pytest.raises(ValueError):
        fit(x, y, sites, d, pr, directory=tmp_path/'part', resume=True, **dict(kw, rank_weight=2.))
