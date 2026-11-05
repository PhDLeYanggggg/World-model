import numpy as np
import pytest
import torch
from src.world_model import m3w_harm_floor_risk as api
from tests.test_m3w_kinematic_risk import fixture
from src.world_model import m3w_admission_priority as setup


def prepared(tmp_path):
    data = fixture()
    data['truth_by_arm']['bound_aware'][1::2, :, 1] = 1.
    settings = dict(width=4, learning_rate=.0001, query_batch_size=2,
        minimum_updates=4, largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    initial = setup.parent.parent.fit(data, arm='mean', seed=17, settings=settings,
        identity={}, path=tmp_path/'original.pt.gz')
    data = setup.parent.prepare(data, initial)
    warm = setup.parent.fit(data, initial, arm='kinematic', settings=settings,
        identity={}, path=tmp_path/'warm.pt.gz')
    data = setup.prepare(data, warm)
    return data, dict(warm, observed_mask_hash=api.parent.kin.digest(data['observed_future'])), settings


def test_weighted_median_and_uniform_control_have_equal_positive_mass():
    x = np.array([.01, .2, 1., -999., 10000.]); w = np.array([1., 4., 1., 1., 0.])
    m = api.positive_weighting(x, w)
    assert m['positive_weighted_median'] == .2
    assert m['floor_positive_mass'] == pytest.approx(m['uniform_multiplier']*m['original_positive_mass'])
    with pytest.raises(ValueError):
        api.positive_weighting(np.zeros(3), np.ones(3))
    target = torch.tensor([[[.02, 0.]], [[.4, 0.]], [[2., 0.]]])
    r = torch.zeros_like(target); d = -torch.ones_like(target)
    row_weights = torch.tensor([1., 4., 1.])
    floor = api.cost(r, d, target, m, 'harm_floor').flatten() @ row_weights
    uniform = api.cost(r, d, target, m, 'uniform_harm').flatten() @ row_weights
    torch.testing.assert_close(floor, uniform)


def test_low_magnitude_harm_gradient_increases_without_changing_safe_or_zero_targets():
    target = torch.tensor([[[.01, 0.]], [[-1., 0.]], [[0., 0.]], [[2., 0.]]])
    r = torch.zeros_like(target, requires_grad=True); d = torch.full_like(r, -.2)
    weights = dict(positive_weighted_median=.2, uniform_multiplier=1.2)
    old = api.cost(r, d, target, weights, 'joint')
    new = api.cost(r, d, target, weights, 'harm_floor')
    a = torch.autograd.grad(old.sum(), r, retain_graph=True)[0]
    b = torch.autograd.grad(new.sum(), r)[0]
    assert b[0].abs().sum() > a[0].abs().sum()
    assert torch.equal(a[1:], b[1:])
    assert old[2].item() == new[2].item() == 0


def test_weighting_ignores_unknown_and_unsupported_targets(tmp_path):
    data, warm, settings = prepared(tmp_path)
    data['observed_future'][0] = False
    a = api.weighting(data, warm, settings)
    data['truth_by_arm']['bound_aware'][0] = 999
    mask = ~data['causal']['allowed']
    data['truth_by_arm']['bound_aware'][mask] = 999
    api.base.exact(a, api.weighting(data, warm, settings))


def test_original_unknown_and_mse_losses_unchanged(tmp_path):
    data, warm, settings = prepared(tmp_path); weights = api.weighting(data, warm, settings)
    data['observed_future'][:] = False
    r = torch.ones(4, 7, 2, requires_grad=True); d = -r
    args = data, np.arange(4), torch.tensor([0, 0, 1, 1]), 2
    original = api.parent.objective((r, d), *args)
    for arm in api.ARMS:
        a = api.objective((r, d), *args, arm, weights)
        assert torch.equal(a['total'], original['total'])
        assert torch.equal(a['risk_MSE'], original['risk_MSE'])


def test_complete_training_control_resume_and_no_future_input(tmp_path):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path); states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(settings=settings, identity={}, arm=arm)
        api.fit(data, warm, **kw, path=path, stop_at=2)
        a = api.fit(data, warm, **kw, path=path, resume=True)
        b = api.fit(data, warm, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.exact_state(a, b); states.append(a)
    old = api.previous.fit(data, warm, arm='joint', settings=settings, identity={}, path=tmp_path/'old.pt.gz')
    api.previous.previous.control_equal(states[0], old)
    api.assert_matched(states)
    before = api.predict(states[-1], data['causal'])
    data['truth_by_arm']['bound_aware'][:] = 999; data['observed_future'][:] = False
    for a, b in zip(before, api.predict(states[-1], data['causal'])):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='causal'):
        api.predict(states[-1], dict(data['causal'], future_endpoint=np.zeros((4, 2))))


def test_failed_screen_never_opens_additional_held_readout(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_moving_harm_floor as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(complete_heads=9, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*args, **kwargs):
        raise AssertionError('No held outcomes when fitting screen fails')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw: None, lambda: None)
    a = json.loads((tmp_path/'analysis.json').read_text())
    assert a['status'] == 'fitting_complete_held_readout_not_run'
    assert a['original_rules_unchanged'] and not a['identity_certification']


def test_incomplete_optimizer_update_preserves_last_atomic_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path)
    kw = dict(settings=settings, identity={}, arm='harm_floor'); path = tmp_path/'state.pt.gz'
    api.fit(data, warm, **kw, path=path, stop_at=2); saved = path.read_bytes()
    original = torch.optim.AdamW.step
    def interrupted(self, *args, **kwargs):
        with torch.no_grad():
            self.param_groups[0]['params'][0].add_(10)
        raise KeyboardInterrupt('incomplete update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.fit(data, warm, **kw, path=path, resume=True)
    assert path.read_bytes() == saved
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    a = api.fit(data, warm, **kw, path=path, resume=True)
    b = api.fit(data, warm, **kw, path=tmp_path/'direct.pt.gz')
    api.parent.exact_state(a, b)
