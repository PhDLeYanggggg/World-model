import numpy as np
import pytest
import torch
from src.world_model import m3w_joint_decision_risk as api
from tests.test_m3w_decoupled_risk import prepared


def test_joint_loss_trains_zero_target_blocking_channel_old_loss_does_not():
    r = torch.tensor([[[-.2, .3]]], requires_grad=True)
    d = torch.tensor([[[-.4, -.1]]], requires_grad=True)
    target = torch.tensor([[[-1., 0.]]])
    old = api.previous.composed_cost(r, d, target).mean()
    before = torch.autograd.grad(old, r, retain_graph=True)[0]
    assert before[0, 0, 1] == 0
    api.joint_cost(r, d, target).sum().backward()
    assert r.grad[0, 0, 1] > 0 and r.grad[0, 0, 0] == 0
    assert torch.count_nonzero(d.grad) == 0


def test_joint_loss_never_cancels_positive_excess_with_other_channel_benefit():
    r = torch.tensor([[[.1, .3]]], requires_grad=True)
    d = -torch.ones_like(r)
    api.joint_cost(r, d, torch.tensor([[[-100., .2]]])).sum().backward()
    assert r.grad[0, 0, 1] < 0
    assert api.joint_cost(r, d, torch.zeros_like(r)).item() == 0


def test_unknown_supervision_and_original_objectives_unchanged(tmp_path):
    data, _, _ = prepared(tmp_path)
    data['observed_future'][0] = False
    r = torch.zeros(4, 7, 2, requires_grad=True)
    d = torch.ones_like(r, requires_grad=True)
    args = (data, np.arange(4), torch.tensor([0, 0, 1, 1]), 2)
    original = api.parent.objective((r, d), *args)
    a = api.objective((r, d), *args, 'per_channel')
    old = api.previous.objective((r, d), *args, 'composed')
    assert torch.equal(a['total'], old['total'])
    b = api.objective((r, d), *args, 'joint')
    assert torch.equal(original['risk_MSE'], b['risk_MSE'])
    assert torch.equal(original['decision_objective'], b['decision_objective'])
    left = torch.autograd.grad(original['total'], (r, d), retain_graph=True)
    right = torch.autograd.grad(b['total'], (r, d))
    for x, y in zip(left, right):
        assert torch.equal(x[~data['observed_future']], y[~data['observed_future']])


def test_real_training_exact_resume_control_and_causal_predict(tmp_path):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path); states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(settings=settings, identity={}, arm=arm)
        api.fit(data, warm, **kw, path=path, stop_at=2)
        a = api.fit(data, warm, **kw, path=path, resume=True)
        b = api.fit(data, warm, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.exact_state(a, b); states.append(a)
    old = api.previous.fit(data, warm, arm='composed', settings=settings, identity={}, path=tmp_path/'old.pt.gz')
    api.previous.control_equal(states[0], old)
    api.assert_matched(states)
    before = api.predict(states[1], data['causal'])
    data['truth_by_arm']['bound_aware'][:] = 999; data['observed_future'][:] = False
    for a, b in zip(before, api.predict(states[1], data['causal'])):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='causal'):
        api.predict(states[1], dict(data['causal'], future_endpoint=np.zeros((4, 2))))


def test_failed_screen_does_not_read_held_data(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_moving_joint_decision as run
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
