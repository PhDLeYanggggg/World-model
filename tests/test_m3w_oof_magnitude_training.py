import numpy as np
import torch
from src.world_model import m3w_oof_magnitude as m
from src.world_model import m3w_membership_auxiliary as neural
from src.world_model.m3w_nested_residual import labels
from src.world_model.m3w_easy_membership_probe import labels as easy_labels


def test_single_reference_real_training_resume_and_archive(tmp_path):
    torch.set_num_threads(4)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(64, 7)).astype(np.float32)
    cv = np.arange(64)/10+.1; raw = np.column_stack([cv, cv/2])
    cv[-1] = np.nan; raw[-1] = np.nan
    sites = np.full(64, 'a'); env = np.full(64, 10.)
    pr = m.reference_preprocess(x, raw, cv, sites, ['b', 'c'])
    y = labels(raw, cv, pr['positive_easy_cut']); easy = easy_labels(cv, pr['positive_easy_cut'])
    settings = dict(steps=10, width=8, batch_size=16, learning_rate=.0003,
        gradient_clip=5, checkpoint_every=2, heartbeat_every=2)
    def fit(home, **kw):
        return neural.fit(x, y, easy, sites, env, pr, arm='cost_only', seed=17,
            settings=settings, identity=dict(test=1), directory=home, heartbeat=lambda **kw:None, **kw)
    a, fa = fit(tmp_path/'a')
    fit(tmp_path/'b', stop_at=4)
    b, fb = fit(tmp_path/'b', resume=True)
    for key in a.state_dict(): assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    assert fa['unknown_rows_sampled'] == fb['unknown_rows_sampled'] == 0
    p, _ = neural.predict(a, x, env, pr)
    m.compress_checkpoint(tmp_path/'a'); restored, state = m.restore(tmp_path/'a')
    np.testing.assert_array_equal(neural.predict(restored, x, env, pr)[0], p)
    assert state['preprocess']['training_sites'] == ['a']


def test_magnitude_readout_respects_units_and_scene_balance():
    p = np.array([[4., 2., 2., 1.], [4., 2., 2., 1.], [2., 1., 1., .5]])
    y = p.copy(); y[:2, 3] *= 2
    env = np.full(3, 10.); sites = np.array(['a', 'a', 'b'])
    head = m.fit_magnitude(p, y, env, sites, 'c')
    scaled = m.fit_magnitude(20*p, 20*y, 20*env, sites, 'c')
    np.testing.assert_allclose(head['slopes'], scaled['slopes'])
    np.testing.assert_allclose(m.predict_magnitude(scaled, 20*p, 20*env),
        20*m.predict_magnitude(head, p, env))
    take = np.array([0, 1, 0, 1, 2])
    duplicate = m.fit_magnitude(p[take], y[take], env[take], sites[take], 'c')
    np.testing.assert_allclose(head['slopes'], duplicate['slopes'])


def test_inner_auxiliary_event_excludes_held_locality(monkeypatch):
    from scripts import run_m3w_european_oof_magnitude as run
    import pytest
    sites = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    v = dict(sites=sites, x=np.arange(12).reshape(6, 2).astype(float),
        env=np.ones(6)*10, outer='d', g=dict(producer_roster=['e']))
    take = sites != 'c'
    target = np.array([[3., 2., 1., 0.], [3., 2., 1., 2.]]*2)
    refs = {s:(s, dict(training_sites=[s]), dict(test=s)) for s in ['a', 'b']}
    def predict(model, x, env, pr):
        return np.tile([3., 1., 1., .5], (len(x), 1)), np.full(len(x), .2)
    monkeypatch.setattr(run.neural, 'predict', predict)
    a = run.inner_event(v, 'c', take, target, refs)
    v['x'][~take] = 1e16; v['env'][~take] = 1e17
    b = run.inner_event(v, 'c', take, target, refs)
    np.testing.assert_array_equal(a[0], b[0]); assert a[1:] == b[1:]
    refs['b'] = ('b', dict(training_sites=['b', 'c']), dict(test='b'))
    with pytest.raises(ValueError): run.inner_event(v, 'c', take, target, refs)
