import inspect

import numpy as np
import pytest
import torch

from src.world_model import m3w_temporal_auxiliary as api


def data():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(24, 4))
    env = np.full(24, 3.)
    y = np.column_stack((.4+.1*np.maximum(x[:, 0], 0),
                         .3+.1*np.maximum(x[:, 1], 0),
                         1.+.1*np.abs(x[:, 2]), np.full(24, .4), np.full(24, .1)))
    v = np.stack([np.column_stack((np.full(12, r[1]-r[0]), np.full(12, r[2]))) for r in y])
    wave = np.sin(np.arange(12))
    v[:, :, 0] += .1*(wave-wave.mean())
    v[0, 6:] = np.nan
    v[0, :6, 0] -= v[0, :6, 0].mean()-(y[0, 1]-y[0, 0])
    y[-1] = np.nan; v[-1] = np.nan
    sites, recordings, frames = np.repeat('site', 24), np.repeat([1, 2, 3], 8), np.tile(np.repeat([0, 1], 4), 3)
    pr = api.core.preprocess(x, env, y, sites, recordings, frames, training_site='site')
    return x, env, y, v, sites, recordings, frames, pr


def settings():
    return dict(width=6, steps=8, learning_rate=.001, query_batch_size=2,
                gradient_clip=5., checkpoint_every=4, heartbeat_every=4, auxiliary_weight=.1)


def fit(tmp_path, arm, name='fit', stop=None, resume=False, args=None):
    return api.fit(*(data() if args is None else args), arm=arm, settings=settings(), seed=17,
                   identity={'registered': 'test'}, path=tmp_path/(name+'.pt.gz'),
                   heartbeat=lambda **kw: None, resume=resume, stop_at=stop)


def test_identical_initial_models_and_future_free_inference():
    pr = data()[-1]
    models = [api.initialize(pr, settings(), 17) for _ in api.ARMS]
    for m in models[1:]: api.core.exact(models[0].state_dict(), m.state_dict())
    assert list(inspect.signature(api.predict).parameters) == ['state', 'x', 'envelope']


def test_control_targets_preserve_primary_mean_and_missing_support():
    args = data(); v = args[3]
    rowmean = api.auxiliary_targets(v, 'rowmean')
    np.testing.assert_array_equal(np.isnan(v), np.isnan(rowmean))
    for i in range(23):
        ok = np.isfinite(v[i, :, 0])
        np.testing.assert_allclose(rowmean[i, ok], np.broadcast_to(v[i, ok].mean(0), (ok.sum(), 2)))
    np.testing.assert_array_equal(api.auxiliary_targets(v, 'temporal'), v)


def test_auxiliary_gradients_mask_unknown_steps_and_do_not_touch_labels():
    pred = torch.ones((2, 12, 2), requires_grad=True)
    y = torch.zeros_like(pred, requires_grad=True)
    mask = torch.ones((2, 12), dtype=torch.bool); mask[0, 4:] = False
    loss = api.temporal.auxiliary_loss(pred, y, mask, torch.tensor([0, 1]), 2)
    loss.backward()
    assert not pred.grad[~mask].any() and y.grad is None


def test_none_arm_is_exact_original_primary_training(tmp_path):
    args = data(); out = fit(tmp_path, 'none')
    x, env, y, _, sites, rec, frames, pr = args
    original = api.core.fit(x, env, y, sites, rec, frames, pr, arm='nonlinear', settings=settings(),
        identity={'registered': 'test'}, seed=17, path=tmp_path/'original.pt.gz', heartbeat=lambda **kw: None)
    model = {k.removeprefix('primary.'):v for k,v in out['model'].items() if k.startswith('primary.')}
    api.core.exact(model, original['model'])
    assert out['draw_hash'] == original['draw_hash']


@pytest.mark.parametrize('arm', api.ARMS)
def test_resume_matches_uninterrupted_training_exactly(tmp_path, arm):
    full = fit(tmp_path, arm, name='full')
    partial = fit(tmp_path, arm, name='resumed', stop=3)
    assert partial['step'] == 3
    resumed = fit(tmp_path, arm, name='resumed', resume=True)
    for key in full:
        if key != 'seconds': api.core.exact(full[key], resumed[key])
    prediction, temporal, support = api.predict(full, data()[0], data()[1])
    assert prediction.shape == (24, 5) and temporal.shape == (24, 12, 2)
    assert np.isfinite(prediction).all() and support.dtype == bool
    assert (prediction[:, :2].sum(1) <= data()[1]+1e-6).all()
    assert full['unknown_rows_sampled'] == 0


def test_three_arms_share_draws_but_auxiliary_reaches_encoder(tmp_path):
    states = [fit(tmp_path, a, name=a) for a in api.ARMS]
    api.assert_matched(states)
    assert not torch.equal(states[0]['model']['primary.encoder.0.weight'], states[2]['model']['primary.encoder.0.weight'])
    assert all(s['trace'][-1]['monitor']['primary_total'] < s['trace'][0]['monitor']['primary_total'] for s in states)


@pytest.mark.parametrize('kind', ['targets', 'features', 'settings', 'identity'])
def test_resume_rejects_changed_training_inputs(tmp_path, kind):
    args = list(data()); fit(tmp_path, 'temporal', stop=3, args=args)
    kwargs = dict(arm='temporal', settings=settings(), seed=17, identity={'registered':'test'},
                  path=tmp_path/'fit.pt.gz', heartbeat=lambda **kw:None, resume=True)
    if kind == 'features': args[0] = args[0].copy(); args[0][0, 0] += .1
    if kind == 'targets':
        args[3] = args[3].copy(); args[3][1, 0, 0] += .1; args[3][1, 1, 0] -= .1
    if kind == 'settings': kwargs['settings']['auxiliary_weight'] = .2
    if kind == 'identity': kwargs['identity']['registered'] = 'changed'
    with pytest.raises((AssertionError, ValueError)): api.fit(*args, **kwargs)


def test_inconsistent_temporal_labels_rejected_before_checkpoint(tmp_path):
    args = list(data()); args[3] = args[3].copy(); args[3][0, :6, 0] += 1
    with pytest.raises(ValueError, match='primary'): fit(tmp_path, 'temporal', args=args)
    assert not (tmp_path/'fit.pt.gz').exists()


def test_storage_reserve_and_run_cap_fail_closed(tmp_path):
    report = api.storage_status(tmp_path, reserve=10, remaining=5, free=14)
    assert not report['allowed'] and report['shortfall_bytes'] == 1
    assert api.storage_status(tmp_path, reserve=10, remaining=5, free=15)['allowed']
    with pytest.raises(OSError): api.require_storage(tmp_path, reserve=10, remaining=5, free=14)


def test_storage_guard_runs_before_model_initialization(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Must not initialize training when storage is blocked')
    monkeypatch.setattr(api, 'initialize', forbidden)
    def blocked():
        raise OSError('storage blocked')
    with pytest.raises(OSError, match='storage blocked'):
        api.fit(*data(), arm='temporal', settings=settings(), seed=17, identity={},
                path=tmp_path/'blocked.pt.gz', heartbeat=lambda **kw:None, checkpoint_guard=blocked)
    assert not (tmp_path/'blocked.pt.gz').exists()


def test_interrupt_retains_atomic_state_and_replays_missing_updates(tmp_path):
    direct = fit(tmp_path, 'temporal', name='direct')
    def interrupt(**kw):
        raise KeyboardInterrupt('synthetic interrupt after an update')
    with pytest.raises(KeyboardInterrupt):
        api.fit(*data(), arm='temporal', settings=settings(), seed=17, identity={'registered':'test'},
                path=tmp_path/'interrupted.pt.gz', heartbeat=interrupt)
    saved = api.core.read_checkpoint(tmp_path/'interrupted.pt.gz')
    assert saved['step'] == 0
    resumed = fit(tmp_path, 'temporal', name='interrupted', resume=True)
    for key in direct:
        if key != 'seconds': api.core.exact(direct[key], resumed[key])
