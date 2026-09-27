import numpy as np
import pytest
import torch
from src.world_model.m3w_risk_excess import excess, loss, fit
from src.world_model.m3w_native_gain_harm import preprocess, cost_loss
from src.world_model import m3w_geometric_cost_head as control


def test_budget_score_not_identified_individual_moments():
    p = torch.tensor([[100., 2.]], requires_grad=True)
    y = torch.tensor([[50., 1.]], requires_grad=True)
    assert loss(p, y).item() == 0
    assert cost_loss(p, y, 'mse').item() > 0
    loss(p, y).backward()
    assert y.grad is None


def test_signed_targets_keep_negative_margin_and_zero_reference_harm():
    np.testing.assert_allclose(excess(np.array([[0., 1.], [50., 0.], [50., 1.]])), [1., -1., 0.])


def test_zero_score_matches_original_float64_screen():
    p = np.array([[7., .14], [50, 1], [0, 1], [2, 0]], np.float32)
    from src.evaluation.m3w_risk_moment_crossfit import screen
    np.testing.assert_array_equal(excess(p) <= 0, screen(p))


def test_budget_and_invalid_inputs_fail_closed():
    with pytest.raises(ValueError): excess(np.ones((2, 2)), budget=.05)
    with pytest.raises(ValueError): excess(np.array([[np.nan, 0]]))
    with pytest.raises(ValueError): loss(torch.ones(2, 2), torch.tensor([[1., -1.], [1., 0.]]))


def test_resume_and_matched_sampler(tmp_path):
    from scripts.replay_m3w_dimensionless_training import exact
    torch.set_num_threads(4)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(48, 5)).astype(np.float32)
    sites = np.repeat(['a', 'b', 'c'], 16)
    cv = np.linspace(.2, 3, 48); cv[[0, 16, 32]] = np.nan
    y = np.column_stack((cv, .1*cv)); env = np.ones(48)
    pr = preprocess(x, y, cv, sites, 'held')
    settings = dict(width=8, steps=8, batch_size=12, learning_rate=.0003, gradient_clip=5,
                    checkpoint_every=4, heartbeat_every=4)
    kwargs = dict(seed=17, settings=settings, identity={'fixed':True}, heartbeat=lambda **kw: None)
    a, ar = fit(x, y, sites, env, pr, directory=tmp_path/'a', **kwargs)
    fit(x, y, sites, env, pr, directory=tmp_path/'b', stop_at=4, **kwargs)
    b, br = fit(x, y, sites, env, pr, directory=tmp_path/'b', resume=True, **kwargs)
    assert ar['unknown_rows_sampled'] == br['unknown_rows_sampled'] == 0
    exact(a.state_dict(), b.state_dict())
    sa = torch.load(tmp_path/'a/checkpoint.pt', weights_only=False)
    sb = torch.load(tmp_path/'b/checkpoint.pt', weights_only=False)
    for k in ('optimizer', 'sampler_rng', 'torch_rng', 'draws', 'trace'): exact(sa[k], sb[k])
    control.fit(x, y, sites, env, pr, task='all', directory=tmp_path/'c', **kwargs)
    sc = torch.load(tmp_path/'c/checkpoint.pt', weights_only=False)
    for k in ('sampler_rng','torch_rng','draws','preprocess','mean_envelope'): exact(sa[k], sc[k])
    assert sa['task'] == 'signed_budget_excess'
    assert sa['settings'] == sc['settings']
    with pytest.raises(ValueError):
        fit(x, y, sites, env, pr, directory=tmp_path/'b', **kwargs)


def test_all_fallback_not_safe_learned_benefit():
    from src.evaluation.m3w_risk_excess import metric
    r = metric([1, 1], [2, 2], [[1, 1], [2, 0]], [2, 1], 0, 1, 1.5, 1.8)
    assert r['new_screen_rate'] == 0
    assert r['new_screen_positive_harm_ratio'] is None
    assert r['new_all_ADE_gain_vs_CV_percent'] == 0


def test_positive_harm_not_net_degradation():
    from src.evaluation.m3w_risk_excess import metric
    r = metric([-1, -1], [1, 1], [[10, 1], [10, 0]], [11, 0], 0, 1, 10, 10)
    assert r['new_screen_positive_harm_ratio'] == .05
    assert r['new_all_ADE_gain_vs_CV_percent'] == pytest.approx(45)
    assert r['new_easy_ADE_gain_vs_CV_percent'] == pytest.approx(45)


def test_zero_reference_and_unknown_are_not_erased():
    from src.evaluation.m3w_risk_excess import metric
    r = metric([-1, 1], [1, 1], [[0, 1], [np.nan, np.nan]], [1, np.nan], 0, 1, 1, 2)
    assert r['rows'] == 1 and r['unknown'] == 1
    assert r['new_zero_reference_harmed'] == 1 and r['new_zero_reference_harm'] == 1
    assert r['new_screen_positive_harm_ratio'] is None
