import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_rollout_history_risk as api
from tests.test_m3w_harm_floor_risk import prepared


def geometry():
    x = np.zeros((2, 7, 428), np.float32); x[..., 24:32] = 1
    h = x[..., :24].reshape(2, 7, 8, 3); h[..., 0] = np.arange(-7, 1)
    floor = x[..., 306:330].reshape(2, 7, 12, 2); floor[..., 0] = np.arange(1, 13)
    candidate = x[..., 330:354].reshape(2, 7, 12, 2); candidate[..., 1] = np.arange(1, 13)
    return x


def test_explicit_causal_geometry_and_static_case():
    x = geometry(); b = api.basis(x)
    assert b.shape == (2, 7, 32) and len(api.NAMES) == 32
    expected = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, -1, 1, -1, 1, 0, 1, 0, 1, 0, 1]
    np.testing.assert_allclose(b[0, 0, :20], expected)
    for i, t in enumerate((1, 3, 6, 12)):
        np.testing.assert_allclose(b[0, 0, 20+3*i:23+3*i], [-t, t, 2**.5*t], rtol=1e-6)
    x[..., :24] = 0; assert np.isfinite(api.basis(x)).all()
    with pytest.raises(ValueError):
        api.basis(x[..., :-1])
    x[..., 24] = 0
    with pytest.raises(ValueError):
        api.basis(x)


def test_rotation_translation_and_unrelated_fields():
    x = geometry(); expected = api.basis(x)
    rotation = np.array([[0., 1.], [-1., 0.]])
    h = x[..., :24].reshape(2, 7, 8, 3); h[..., :2] = h[..., :2]@rotation+35
    for s in (306, 330):
        p = x[..., s:s+24].reshape(2, 7, 12, 2); p[:] = p@rotation
    x[..., 32:306] = 123; x[..., 354:] = -123
    np.testing.assert_allclose(api.basis(x), expected, atol=1e-6)


def test_initial_identity_actual_training_resume_and_control(tmp_path):
    torch.set_num_threads(1); d, origin, settings = prepared(tmp_path); d = api.prepare(d, origin)
    old = api.parent.initialize(origin, 'separate64'); new = api.initialize(origin)
    seg = torch.tensor([0, 0, 1, 1]); common = api.parent.kin.batch_features(d, 'kinematic')
    for arm in api.ARMS:
        z = api.parent.kin.batch_features(api.extended(d, arm), 'kinematic')
        api.base.exact(old(common, seg, 2), new(z, seg, 2))
    states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(arm=arm, settings=settings, identity={})
        api.fit(d, origin, **kw, path=path, stop_at=2)
        a = api.fit(d, origin, **kw, path=path, resume=True)
        b = api.fit(d, origin, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.exact_state(a, b); states.append(a)
        assert bool(a['model']['risk.relation.weight'].any()) == (arm == 'relational')
        assert bool(a['model']['decision.relation.weight'].any()) == (arm == 'relational')
    api.assert_matched(states)
    original = api.previous.fit(d, origin, arm='joint', settings=settings, identity={}, path=tmp_path/'old.pt.gz')
    for key, value in original['model'].items():
        api.base.exact(states[0]['model'][key], value)
    api.base.exact(api.full(states[0], d), api.parent.full(api.previous.model_from(original), d, original['sampler']['counts']))


def test_causal_prediction_future_mask_and_resume_binding(tmp_path):
    torch.set_num_threads(1); d, origin, settings = prepared(tmp_path); d = api.prepare(d, origin)
    kw = dict(arm='relational', settings=settings, identity={}, path=tmp_path/'fit.pt.gz')
    state = api.fit(d, origin, **kw); before = api.predict(state, d['causal'])
    changed = copy.deepcopy(d); changed['truth_by_arm']['bound_aware'][:] = 1e9
    changed['observed_future'][:] = False
    api.base.exact(before, api.predict(state, changed['causal']))
    with pytest.raises(ValueError, match='causal'):
        api.predict(state, dict(d['causal'], future_endpoint=np.zeros((4, 2))))
    with pytest.raises(AssertionError):
        api.fit(changed, origin, **kw, resume=True)
    with pytest.raises(AssertionError):
        api.fit(d, origin, **dict(kw, settings=dict(settings, learning_rate=.001)), resume=True)


def test_partial_optimizer_failure_preserves_completed_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1); d, origin, settings = prepared(tmp_path); d = api.prepare(d, origin)
    path = tmp_path/'fit.pt.gz'; kw = dict(arm='relational', settings=settings, identity={}, path=path)
    api.fit(d, origin, **kw, stop_at=2); saved = path.read_bytes()
    def fail(optimizer, *a, **k):
        with torch.no_grad():
            optimizer.param_groups[0]['params'][0].add_(1.)
        raise RuntimeError('partial update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError, match='partial update'):
        api.fit(d, origin, **kw, resume=True)
    assert path.read_bytes() == saved
