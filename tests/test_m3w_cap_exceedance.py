import numpy as np
import pytest
import torch
from src.world_model import m3w_cap_exceedance as m
from src.evaluation import m3w_cap_exceedance as e


def fixture():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(96, 6)); x[::7, 1] = np.nan
    y = (x[:, 0] > .5).astype(float); y[::9] = np.nan
    sites = np.repeat(['a', 'b', 'c'], 32)
    return x, y, sites


def test_cap_event_is_not_any_harm_and_unknown_stays_unknown():
    p = np.tile([1., .5, .3, .2], (4, 1))
    y = np.array([[1, .1, .1, .1], [1, 1, 1, .8], [0, 0, 0, 0], [np.nan]*4])
    np.testing.assert_equal(m.event_target(y, p, np.ones(4)), [0, 1, 0, np.nan])
    p[2] = 0
    np.testing.assert_equal(m.event_target(y, p, np.array([1, 1, 0, 1])), [0, 1, np.nan, np.nan])


def test_illegal_partial_labels_rejected():
    with pytest.raises(ValueError):
        m.event_target(np.array([[1, np.nan, 0, 0]]), np.array([[1, .5, .2, .1]]), np.ones(1))


def test_causal_feature_missingness_and_shapes():
    native = np.zeros((2, 3)); context = np.zeros((2, 7)); context[1, 2] = np.nan
    p = np.tile([1., .5, .3, .2], (2, 1))
    x = m.causal_features(native, context, p, np.ones(2))
    assert x.shape == (2, 19) and x[1, 12] == 1
    with pytest.raises(ValueError):
        m.causal_features(native, context[:, :2], p, np.ones(2))


def test_preprocess_excludes_unknown_and_outer():
    x, y, s = fixture(); pr = m.preprocess(x, y, s, 'd')
    bad = x.copy(); bad[~np.isfinite(y)] = 1e8
    pr2 = m.preprocess(bad, y, s, 'd')
    for k in ('mean', 'std', 'weights'):
        np.testing.assert_equal(pr[k], pr2[k])
    assert np.isfinite(m.transform(x, pr)).all() and np.max(abs(m.transform(x, pr))) <= 20
    for site in set(s):
        assert pr['weights'][s == site].sum() == pytest.approx(1/3)
    with pytest.raises(ValueError): m.preprocess(x, y, s, 'a')
    with pytest.raises(ValueError): m.preprocess(x, np.zeros(len(y)), s, 'd')


@pytest.mark.parametrize('arm', m.ARMS)
def test_exact_checkpoint_resume_and_identity(tmp_path, arm):
    torch.set_num_threads(4)
    x, y, s = fixture()
    settings = dict(width=8, steps=24, batch_size=16, learning_rate=.001,
                    gradient_clip=5, checkpoint_every=8, heartbeat_every=8)
    kw = dict(arm=arm, seed=7, settings=settings, identity={'frozen': 'abc'}, heartbeat=lambda **kw: None)
    full, pr, fit = m.fit(x, y, s, 'd', directory=tmp_path/'full', **kw)
    m.fit(x, y, s, 'd', directory=tmp_path/'resumed', stop_at=8, **kw)
    model, pr2, resumed = m.fit(x, y, s, 'd', directory=tmp_path/'resumed', resume=True, **kw)
    np.testing.assert_equal(m.predict(full, x, pr), m.predict(model, x, pr2))
    restored, state = m.restore(tmp_path/'resumed')
    np.testing.assert_equal(m.predict(restored, x, state['preprocess']), m.predict(model, x, pr2))
    assert fit['complete'] and resumed['complete'] and not fit['unknown_rows_sampled']
    with pytest.raises(ValueError):
        m.fit(x, y, s, 'd', directory=tmp_path/'resumed', **kw)
    kw['identity'] = {'frozen': 'changed'}
    with pytest.raises(ValueError):
        m.fit(x, y, s, 'd', directory=tmp_path/'resumed', resume=True, **kw)


def test_support_counts_tracks_not_windows():
    target = np.array([1, 1, 0, np.nan]); site = np.repeat('a', 4)
    row = e.support(target, site, np.array([2, 2, 3, 4]), np.array([7, 7, 7, 8]))[0]
    assert row['positive'] == 2 and row['positive_agents'] == 1 and row['agents'] == 2


def test_probabilistic_metrics_and_missing_class():
    y = np.array([0, 0, 1, 1, np.nan]); p = np.array([.1, .2, .8, .9, .5])
    row = e.measures(y, p, np.array([0, 0, 1, 2, np.nan]), .5)
    assert row['AUROC'] == 1 and row['AP'] == 1 and row['BCE'] < row['prior_BCE']
    assert row['top10_overshoot_mass'] == pytest.approx(.8/3)
    one = e.measures(np.zeros(4), np.ones(4)*.1, np.zeros(4), .1)
    assert one['AUROC'] is None and one['AP'] is None and one['top10_overshoot_mass'] is None
    with pytest.raises(ValueError): e.measures(y, p+2, np.zeros(5), .5)


def test_locality_bootstrap_and_missing_views():
    a = np.tile([1, 2, 3, 4.], (3, 1))
    r = e.paired_interval(a, seed=1, draws=3000)
    assert r['point'] == 2.5 and r['low'] > 0 and r['seeds'] == 3 and r['localities'] == 4
    a[0, 0] = np.nan
    assert e.paired_interval(a, seed=1, draws=3000)['status'] == 'not_estimable'
