import numpy as np
import torch
import pytest
from src.world_model.m3w_bounded_cost_head import ARMS, build, fit, predict, loss_value
from src.world_model.m3w_native_gain_harm import preprocess


def test_identical_initialization_and_capacity():
    models = [build(5, 8, 17) for _ in ARMS]
    for model in models[1:]:
        for a, b in zip(models[0].parameters(), model.parameters()):
            assert torch.equal(a, b)


@pytest.mark.parametrize('arm', ARMS)
def test_bound_zero_and_gradients(arm):
    model = build(5, 8, 3)
    x = torch.arange(35, dtype=torch.float32).reshape(7, 5)/10
    d = torch.tensor([0., .001, .1, 1., 4., 100., 10000.])
    p = model(x, d, arm)
    assert torch.equal(p[0], torch.zeros(2)) and (p >= 0).all()
    if arm != 'direct_native':
        assert (p.sum(1) <= d).all()
    loss = loss_value(p, torch.zeros_like(p), d, arm)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters())


@pytest.mark.parametrize('arm', ARMS)
def test_resume_matches_uninterrupted_and_unknowns_never_sampled(tmp_path, arm):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    d = rng.uniform(.5, 5, 40)
    y = d[:, None]*rng.uniform(0, .4, (40, 2))
    sites = np.repeat(['a', 'b'], 20)
    cv = np.ones(40)*3
    y[0], cv[0] = np.nan, np.nan
    pr = preprocess(x, y, cv, sites, 'c')
    settings = dict(width=8, learning_rate=.001, steps=12, batch_size=16, gradient_clip=5., heartbeat_every=4, checkpoint_every=4)
    kw = dict(arm=arm, seed=17, settings=settings, identity={'synthetic':True}, heartbeat=lambda **x:None)
    full, f = fit(x, y, d, sites, pr, directory=tmp_path/'full', **kw)
    fit(x, y, d, sites, pr, directory=tmp_path/'resume', stop_at=4, **kw)
    resumed, r = fit(x, y, d, sites, pr, directory=tmp_path/'resume', resume=True, **kw)
    np.testing.assert_array_equal(predict(full, x, d, pr, arm), predict(resumed, x, d, pr, arm))
    assert f['complete'] and r['complete'] and f['unknown_rows_sampled'] == r['unknown_rows_sampled'] == 0


def test_fraction_loss_changes_weighting_not_labels():
    score = torch.tensor([[1., 0.], [10., 0.]])
    target = torch.zeros_like(score)
    d = torch.tensor([1., 10.])
    assert float(loss_value(score, target, d, 'direct_native')) == 25.25
    assert float(loss_value(score, target, d, 'bounded_fraction')) == .5


def test_past_only_policy_support_and_exact_count():
    from scripts.run_m3w_bounded_cost import selections
    past = np.zeros((4, 8, 2))
    past[1:, -1, 0] = 1
    cost = np.array([[100., 0.], [10., .1], [5., 4.], [1., 2.]])
    out = selections(cost, past, np.array([1., 1., 0., 1.]), np.array([False, True, False, False]), np.arange(4))
    np.testing.assert_array_equal(out['strict_stop'], [False, True, False, False])
    np.testing.assert_array_equal(out['matched_count'], [False, True, False, False])


def test_resume_rejects_changed_identity(tmp_path):
    x, d, sites = np.zeros((4, 2), np.float32), np.ones(4), np.array(['a', 'a', 'b', 'b'])
    y = np.zeros((4, 2))
    pr = preprocess(x, y, np.ones(4), sites, 'c')
    settings = dict(width=4, learning_rate=.001, steps=2, batch_size=4, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    kw = dict(arm='direct_native', seed=17, settings=settings, directory=tmp_path, heartbeat=lambda **x:None)
    fit(x, y, d, sites, pr, identity={'id':1}, **kw)
    with pytest.raises(ValueError, match='identity'):
        fit(x, y, d, sites, pr, identity={'id':2}, resume=True, **kw)
