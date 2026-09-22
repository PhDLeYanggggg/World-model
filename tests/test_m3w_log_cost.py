import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.world_model.m3w_log_cost_head import composition, log_composition, loss_value, fit
from src.world_model.m3w_conditional_cost_head import fit as reference_fit, region_weights
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_native_gain_harm import preprocess


def test_same_forward_parameterization():
    logits = torch.tensor([[-100., 2.], [-20., 1.], [0., 0.], [90., -40.]], requires_grad=True)
    lp = log_composition(logits)
    positive = torch.nn.functional.softplus(logits)
    torch.testing.assert_close(lp.exp()[:, :2], positive / (1 + positive.sum(1, keepdim=True)))
    (-lp.sum()).backward()
    assert torch.isfinite(logits.grad).all()


def test_simplex_and_no_target_weight_or_scale_gradient():
    z = torch.zeros((3, 2), requires_grad=True)
    y = torch.tensor([[1., 0.], [0., .3], [0., 0.]], requires_grad=True)
    d = torch.tensor([1., 1., 0.], requires_grad=True)
    w = torch.ones(3, requires_grad=True)
    torch.testing.assert_close(composition(y, d), torch.tensor([[1., 0., 0.], [0., .3, .7], [0., 0., 1.]]))
    loss_value(z, y, d, w).backward()
    assert y.grad is None and w.grad is None and d.grad is None
    assert torch.isfinite(z.grad).all() and (z.grad[2] == 0).all()


@pytest.mark.parametrize('y,d', [([[1., 1.]], [1.]), ([[.1, 0]], [0.]),
    ([[-1., 0]], [1.]), ([[float('nan'), 0]], [1.])])
def test_invalid_supervision_rejected(y, d):
    with pytest.raises(ValueError):
        composition(torch.tensor(y), torch.tensor(d))


def test_loss_matches_numpy_soft_composition():
    z = torch.tensor([[1., -2.], [.4, 2.]], dtype=torch.float64)
    y = torch.tensor([[.4, 0.], [0., .3]], dtype=torch.float64)
    d = torch.tensor([2., 1.], dtype=torch.float64)
    w = torch.tensor([.5, 2.], dtype=torch.float64)
    p = np.logaddexp(0, z.numpy()); p = np.c_[p, np.ones(2)]; p /= p.sum(1, keepdims=True)
    q = y.numpy() / d.numpy()[:, None]; q = np.c_[q, 1 - q.sum(1)]
    expected = np.mean(w.numpy() * d.numpy() * -(q * np.log(p)).sum(1))
    assert float(loss_value(z, y, d, w)) == pytest.approx(expected)


def test_log_loss_corrects_near_zero_harm_in_synthetic_case():
    z = torch.tensor([[2., -20.]], requires_grad=True)
    y = torch.tensor([[0., .25]]); d = torch.ones(1); w = torch.ones(1)
    log_grad = torch.autograd.grad(loss_value(z, y, d, w), z)[0][0, 1]
    score = log_composition(z).exp()[:, :2]
    square_grad = torch.autograd.grad((score - y).square().mean(), z)[0][0, 1]
    assert log_grad < -.2 and abs(square_grad) < 1e-7


def test_same_budget_draws_and_exact_resume(tmp_path):
    rng = np.random.default_rng(38202)
    x = rng.normal(size=(40, 5)).astype(np.float32); d = rng.uniform(.1, 5, 40)
    y = d[:, None] * rng.uniform(0, .3, (40, 2)); cv = np.ones(40)
    sites = np.repeat(['a', 'b'], 20); y[0], cv[0] = np.nan, np.nan
    pr = preprocess(x, y, cv, sites, 'c'); w = region_weights(np.arange(40) % 3 == 0, pr)
    settings = dict(width=8, learning_rate=.001, steps=12, batch_size=16,
                    gradient_clip=5., heartbeat_every=4, checkpoint_every=4)
    kw = dict(seed=17, settings=settings, identity={'synthetic': True}, heartbeat=lambda **_: None)
    model, r = fit(x, y, d, sites, pr, w, directory=tmp_path/'full', **kw)
    fit(x, y, d, sites, pr, w, directory=tmp_path/'resumed', stop_at=4, **kw)
    other, rr = fit(x, y, d, sites, pr, w, directory=tmp_path/'resumed', resume=True, **kw)
    reference_fit(x, y, d, sites, pr, w, directory=tmp_path/'reference', **kw)
    np.testing.assert_array_equal(predict(model, x, d, pr, 'bounded_native'), predict(other, x, d, pr, 'bounded_native'))
    a = torch.load(tmp_path/'full/checkpoint.pt', weights_only=False)
    b = torch.load(tmp_path/'reference/checkpoint.pt', weights_only=False)
    np.testing.assert_array_equal(a['draws'], b['draws'])
    assert r['complete'] and rr['complete'] and not r['unknown_rows_sampled']
    bad = w.copy(); bad[1] *= 2
    with pytest.raises(AssertionError):
        fit(x, y, d, sites, pr, bad, directory=tmp_path/'resumed', resume=True, **kw)


def test_config_does_not_change_roles_budget_or_policies():
    from scripts.run_m3w_log_cost import validate_config
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/m3w_log_cost_v1.json').read_text())
    old = json.loads((root/'configs/m3w_adaptive_region_cost_v1.json').read_text())
    validate_config(cfg, old)
    for key, value in [('threshold_search', True), ('primary_reference', 'other'), ('region_multiplier', 9.)]:
        with pytest.raises(ValueError):
            validate_config(dict(cfg, **{key: value}), old)
