import numpy as np
import pytest
import torch
from src.world_model.m3w_selected_risk_learning import event_targets, EventMomentHead, objective, decisions, matched_hash, scalar_decisions, fit


def test_nested_target_event_is_CV_not_reference_and_unknown_stays_unknown():
    y = event_targets(np.array([1., 3., np.nan]), np.array([10., 1., np.nan]),
                      np.array([11., 3., np.nan]), 2.)
    np.testing.assert_array_equal(y[:2], [[10, 1, 10, 1], [1, 2, 0, 0]])
    assert np.isnan(y[2]).all()


def test_event_head_respects_nested_moments_and_zero_envelope():
    m = EventMomentHead(3, 4)(torch.randn(7, 3), torch.arange(7).float())
    assert torch.isfinite(m).all() and (m[:, 2:] <= m[:, :2]).all()
    assert (m[:, 1] <= torch.arange(7)).all() and m[0, 1] == m[0, 3] == 0


def test_selected_group_loss_detects_bias_even_if_population_mean_cancels():
    p = torch.tensor([[2., 1., 1., .5], [0., 0., 0., 0.]], requires_grad=True)
    y = torch.tensor([[1., .5, .5, .25], [1., .5, .5, .25]])
    masks = torch.tensor([[True, True, True], [True, False, False]])
    args = (p, y, torch.ones(4), masks, np.array(['a', 'a']))
    mean, _ = objective(*args, 'mean'); selected, info = objective(*args, 'selected')
    assert selected > mean and info['nonempty_groups'] == 3
    selected.backward(); assert torch.isfinite(p.grad).all()


def test_query_budget_can_select_subset_without_future_support():
    u = np.array([[3., 0.], [2., 0.], [0., 1.]])
    m = np.array([[10., .45, 10., .45], [10., .45, 10., .45], [10., 0., 10., 0.]])
    args = (u, m, np.ones(3, bool), np.ones(3), [np.arange(3)], np.arange(3))
    assert not decisions(*args, 'dual').any()
    assert not decisions(*args, 'scene').any()
    q = decisions(*args, 'joint'); np.testing.assert_array_equal(q, [True, False, False])
    h = matched_hash(u, args[2], args[3], args[4], args[5], q); assert h.sum() == q.sum()


def test_future_support_not_in_decision_api_and_missing_query_rejected():
    u = np.ones((2, 2)); m = np.ones((2, 4))
    with pytest.raises(ValueError): decisions(u, m, np.ones(2, bool), np.ones(2), [np.array([0])], np.arange(2), 'joint')
    with pytest.raises(TypeError): decisions(u, m, np.ones(2, bool), np.ones(2), [np.arange(2)], np.arange(2), 'joint', future=np.ones(2))


def test_scalar_decisions_and_query_budgets():
    rng = np.random.default_rng(9); n = 87
    u = rng.random((n, 2)); m = rng.random((n, 4)); m[:, (0, 2)] *= 40
    env = rng.random(n); moving = rng.random(n) > .2; queries = np.array_split(np.arange(n), 11)
    for mode in ('all', 'dual', 'scene', 'joint'):
        a = decisions(u, m, moving, env, queries, np.arange(n), mode)
        np.testing.assert_array_equal(a, scalar_decisions(u, m, moving, env, queries, np.arange(n), mode))
        if mode in ('scene', 'joint'):
            for q in queries:
                assert (m[q[a[q]]][:, [1, 3]].sum(0) <= .02*m[q][:, [0, 2]].sum(0)).all()


def test_exact_training_resume_and_unknown_labels_not_sampled(tmp_path):
    from src.world_model.m3w_native_gain_harm import preprocess
    rng = np.random.default_rng(4); x = rng.normal(size=(40, 3)).astype(np.float32)
    cv = np.linspace(.5, 3, 40); r = np.ones(40); p = r+np.linspace(0, .9, 40)
    cv[-1] = r[-1] = p[-1] = np.nan
    y = event_targets(cv, r, p, 1.5); sites = np.array(['a']*20+['b']*20)
    pr = preprocess(x, y[:, :2], cv, sites, 'held')
    settings = dict(width=4, steps=6, batch_size=8, learning_rate=.001, gradient_clip=5., heartbeat_every=3, checkpoint_every=3)
    args = (x, y, sites, np.ones(40), np.ones((40, 3), bool), pr)
    kwargs = dict(arm='selected', seed=7, settings=settings, identity={'synthetic':True}, heartbeat=lambda **v:None)
    a, fa = fit(*args, **kwargs, directory=tmp_path/'full')
    fit(*args, **kwargs, directory=tmp_path/'resume', stop_at=3)
    b, fb = fit(*args, **kwargs, directory=tmp_path/'resume', resume=True)
    for k,v in a.state_dict().items(): torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
    assert fa['unknown_rows_sampled'] == fb['unknown_rows_sampled'] == 0
    assert fa['trace'][-1]['loss'] == fb['trace'][-1]['loss']
