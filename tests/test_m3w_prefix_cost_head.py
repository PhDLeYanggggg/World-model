import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.world_model.m3w_prefix_cost_head import build, fit, predict, arm_arrays, loss_value
from src.world_model.m3w_bounded_cost_head import build as scalar_build
from src.world_model.m3w_log_cost_head import loss_value as scalar_loss
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_conditional_cost_head import region_weights
from src.evaluation.m3w_prefix_cost_policy import selections


def test_initialization_capacity_and_bounds():
    a, b = build(356, 128, 17), build(356, 128, 17)
    scalar = scalar_build(356, 128, 17)
    torch.testing.assert_close(a.network[0].weight, scalar.network[0].weight, rtol=0, atol=0)
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0, atol=0)
    assert sum(p.numel() for p in a.parameters()) == 48792
    d = torch.rand(4, 12); d[0] = 0
    score = a(torch.randn(4, 356), d)
    assert (score >= 0).all() and (score.sum(-1) <= d).all() and (score[0] == 0).all()


def test_repeated_tasks_equal_scalar_objective():
    z = torch.tensor([[1., -2.], [-30., 2.]], requires_grad=True)
    y = torch.tensor([[.4, 0.], [0., .3]], requires_grad=True)
    d = torch.tensor([2., 1.], requires_grad=True); w = torch.ones(2, requires_grad=True)
    loss = loss_value(z[:, None].repeat(1, 12, 1), y[:, None].repeat(1, 12, 1), d[:, None].repeat(1, 12), w)
    torch.testing.assert_close(loss, scalar_loss(z, y, d, w))
    loss.backward()
    assert torch.isfinite(z.grad).all() and y.grad is None and d.grad is None and w.grad is None


def test_arm_changes_only_supervision_profile():
    y = np.arange(72).reshape(3, 12, 2); d = np.ones((3, 12))
    a, b = arm_arrays(y, d, 'terminal_repeat')
    np.testing.assert_array_equal(a, np.repeat(y[:, -1:], 12, 1))
    assert arm_arrays(y, d, 'prefix')[0] is y
    with pytest.raises(ValueError):
        arm_arrays(y, d, 'other')


def test_causal_guard_and_equal_count():
    n = 8; ids = np.arange(n); past = np.zeros((n, 8, 2)); past[:, -1, 0] = 1
    d = np.ones((n, 12)); c = np.tile([1., .01], (n, 12, 1)); p = c.copy()
    p[0, 0] = [1., .5]; p[1, -1] = [.01, 1.]; past[2] = 0; d[3] = 0
    p[4, 0] = [0, 0]
    out = selections(c, p, past, d, ids)
    assert out['profile_terminal'][0] and not out['profile_guard'][0]
    assert out['profile_guard'][4] and not out['profile_guard'][1:4].any()
    assert out['control_matched'].sum() == out['profile_guard'].sum() == out['profile_matched'].sum()
    assert not out['control_matched'][2:4].any()
    # Inference has no target, validity, actual easy label or observed prefix length.
    import inspect
    assert list(inspect.signature(selections).parameters) == ['control', 'profile', 'past', 'distance', 'ids']
    for bad in (np.full_like(p, np.nan), -p):
        with pytest.raises(ValueError):
            selections(c, bad, past, d, ids)


@pytest.mark.parametrize('arm', ['terminal_repeat', 'prefix'])
def test_resume_and_same_draws_across_arms(tmp_path, arm):
    rng = np.random.default_rng(4403)
    x = rng.normal(size=(40, 5)).astype(np.float32); d = rng.uniform(.1, 5, (40, 12))
    y = d[..., None] * rng.uniform(0, .3, (40, 12, 2)); cv = np.ones(40)
    sites = np.repeat(['a', 'b'], 20); y[0], cv[0] = np.nan, np.nan
    pr = preprocess(x, y[:, -1], cv, sites, 'c'); w = region_weights(np.arange(40) % 3 == 0, pr)
    settings = dict(width=8, learning_rate=.001, steps=12, batch_size=16, gradient_clip=5., heartbeat_every=4, checkpoint_every=4)
    kw = dict(seed=17, settings=settings, identity={'synthetic': True}, heartbeat=lambda **_: None)
    model, r = fit(x, y, d, sites, pr, w, arm=arm, directory=tmp_path/'full', **kw)
    fit(x, y, d, sites, pr, w, arm=arm, directory=tmp_path/'resume', stop_at=4, **kw)
    other, rr = fit(x, y, d, sites, pr, w, arm=arm, directory=tmp_path/'resume', resume=True, **kw)
    np.testing.assert_array_equal(predict(model, x, d, pr), predict(other, x, d, pr))
    opposite = 'prefix' if arm == 'terminal_repeat' else 'terminal_repeat'
    fit(x, y, d, sites, pr, w, arm=opposite, directory=tmp_path/'other', **kw)
    a = torch.load(tmp_path/'full/checkpoint.pt', weights_only=False)
    b = torch.load(tmp_path/'other/checkpoint.pt', weights_only=False)
    np.testing.assert_array_equal(a['draws'], b['draws'])
    assert r['complete'] and rr['complete'] and not r['unknown_rows_sampled']
    with pytest.raises(ValueError):
        fit(x, y, d, sites, pr, w, arm=opposite, directory=tmp_path/'resume', resume=True, **kw)


def test_frozen_config():
    from scripts.run_m3w_prefix_cost import validate_config
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/m3w_prefix_cost_v1.json').read_text())
    old = json.loads((root/'configs/m3w_log_cost_v1.json').read_text())
    validate_config(cfg, old)
    for key, value in [('threshold_search', True), ('primary_policy', 'profile_terminal'), ('region_multiplier', 9.)]:
        with pytest.raises(ValueError):
            validate_config(dict(cfg, **{key: value}), old)
