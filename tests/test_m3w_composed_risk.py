import numpy as np
import pytest
import torch
from src.world_model import m3w_composed_risk as api
from tests.test_m3w_decoupled_risk import prepared


def test_composed_cost_updates_the_blocking_head():
    r = torch.tensor([.3, -.2], requires_grad=True)
    d = torch.tensor([-.4, .5], requires_grad=True)
    y = torch.tensor([-1., 1.])
    loss = api.composed_cost(r, d, y).sum(); loss.backward()
    assert r.grad[0] > 0 and d.grad[0] == 0
    assert r.grad[1] == 0 and d.grad[1] < 0


def test_unknown_bounds_and_original_loss_remain_unchanged(tmp_path):
    data, warm, _ = prepared(tmp_path)
    data['observed_future'][np.flatnonzero(data['causal']['allowed'].any(1))[0]] = False
    ix = np.arange(4); seg = torch.tensor([0, 0, 1, 1])
    r = torch.zeros(4, 7, 2, requires_grad=True); d = torch.ones_like(r, requires_grad=True)
    a = api.objective((r, d), data, ix, seg, 2, 'independent')
    original = api.parent.objective((r, d), data, ix, seg, 2)
    assert torch.equal(a['total'], original['total'])
    b = api.objective((r, d), data, ix, seg, 2, 'composed')
    ga = torch.autograd.grad(a['total'], (r, d), retain_graph=True)
    gb = torch.autograd.grad(b['total'], (r, d))
    unknown = (~data['observed_future'])[:, None] & data['causal']['allowed']
    for left, right in zip(ga, gb):
        assert torch.equal(left[unknown], right[unknown])
    assert b['composed_observed_cost'] > 0


def test_real_fit_exact_resume_and_control_reproduction(tmp_path):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path)
    states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(settings=settings, identity={}, arm=arm)
        api.fit(data, warm, **kw, path=path, stop_at=2)
        a = api.fit(data, warm, **kw, path=path, resume=True)
        b = api.fit(data, warm, **kw, path=tmp_path/(arm+'_repeat.pt.gz'))
        api.parent.exact_state(a, b)
        for x, y in zip(api.predict(a, data['causal']), api.predict(b, data['causal'])):
            np.testing.assert_array_equal(x, y)
        states.append(a)
    control = api.parent.fit(data, warm, arm='separate64', settings=settings, identity={}, path=tmp_path/'old.pt.gz')
    api.control_equal(states[0], control)
    api.assert_matched(states)
    before = api.predict(states[-1], data['causal'])
    data['truth_by_arm']['bound_aware'][:] = 999; data['observed_future'][:] = False
    for x, y in zip(before, api.predict(states[-1], data['causal'])):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match='causal'):
        api.predict(states[-1], dict(data['causal'], future_endpoint=np.zeros((4, 2))))


def test_zero_observed_rows_add_no_loss(tmp_path):
    data, _, _ = prepared(tmp_path); data['observed_future'][:] = False
    r = torch.ones(4, 7, 2, requires_grad=True); d = -r
    kw = dict(data=data, ix=np.arange(4), segments=torch.tensor([0, 0, 1, 1]), count=2)
    a = api.objective((r, d), **kw, arm='independent')
    b = api.objective((r, d), **kw, arm='composed')
    assert b['composed_observed_cost'] == 0
    assert torch.equal(a['total'], b['total'])


def test_failed_fitting_screen_does_not_read_held_data(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_moving_composed_risk as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(complete_heads=6, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*args, **kwargs):
        raise AssertionError('No additional held readout permitted')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw: None, lambda: None)
    a = json.loads((tmp_path/'analysis.json').read_text())
    assert a['status'] == 'fitting_complete_held_readout_not_run'
    assert a['original_rules_unchanged'] and not a['identity_certification']


def test_interruption_does_not_save_half_updated_parameters(tmp_path, monkeypatch):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path)
    kw = dict(settings=settings, identity={}, arm='composed'); path = tmp_path/'state.pt.gz'
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
