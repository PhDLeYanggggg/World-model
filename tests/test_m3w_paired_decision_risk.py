import numpy as np
import pytest
import torch
from tests.test_m3w_harm_floor_risk import prepared
from src.world_model import m3w_paired_decision_risk as api


def test_null_value_and_gradient_matched_but_pair_translation_invariant():
    p, n = torch.zeros(3, requires_grad=True), torch.zeros(3, requires_grad=True)
    weight = torch.tensor([.1, 1., 2.])
    a = api.contrast(p, n, weight, 'paired_point')
    b = api.contrast(p, n, weight, 'paired_rank')
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ag = torch.autograd.grad(a.sum(), (p, n), retain_graph=True)
    bg = torch.autograd.grad(b.sum(), (p, n))
    for x, y in zip(ag, bg):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(api.contrast(p+3, n+3, weight, 'paired_rank'), b)


def test_pairs_stay_in_query_and_exclude_zero_unknown_and_unproposed():
    y = torch.zeros(8, 7, 2); y[:, 0, 0] = torch.tensor([1., -1., 2., -3., 5., -5., 0., -10.])
    allowed = torch.zeros(8, 7, dtype=torch.bool); allowed[:, 0] = True
    observed = torch.ones(8, dtype=torch.bool); observed[-1] = False
    segments = torch.tensor([0, 0, 1, 1, 2, 3, 0, 0])
    found = api.pairs(y, allowed, observed, segments, 4, 0)
    assert len(found) == 2
    for group, p, n, w in found:
        assert (segments[p[:, 0]] == group).all() and (segments[n[:, 0]] == group).all()
        assert (w > 0).all() and (p[:, 1] == 0).all() and (n[:, 1] == 0).all()
    assert api.pairs(y, allowed, torch.zeros_like(observed), segments, 4, 0) == []


def test_empty_contrast_leaves_unknown_supervision_unchanged(tmp_path):
    data, warm, settings = prepared(tmp_path); data['observed_future'][:] = False
    r = torch.ones(4, 7, 2, requires_grad=True); d = -r
    args = data, np.arange(4), torch.tensor([0, 0, 1, 1]), 2
    original = api.previous.objective((r, d), *args, 'joint')
    for arm in api.ARMS:
        actual = api.objective((r, d), *args, arm, 0)
        assert torch.equal(actual['total'], original['total'])
        assert torch.equal(actual['risk_MSE'], original['risk_MSE'])


def test_three_arm_exact_resume_control_and_inference_is_causal(tmp_path):
    torch.set_num_threads(1); data, warm, settings = prepared(tmp_path); states = []
    for arm in api.ARMS:
        kw = dict(arm=arm, settings=settings, identity={}); path = tmp_path/(arm+'.pt.gz')
        api.fit(data, warm, **kw, path=path, stop_at=2)
        a = api.fit(data, warm, **kw, path=path, resume=True)
        b = api.fit(data, warm, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.exact_state(a, b); states.append(a)
    old = api.previous.fit(data, warm, arm='joint', settings=settings, identity={}, path=tmp_path/'old.pt.gz')
    api.previous.previous.control_equal(states[0], old); api.assert_matched(states)
    before = api.predict(states[-1], data['causal'])
    data['truth_by_arm']['bound_aware'][:] = 1e5; data['observed_future'][:] = False
    for x, y in zip(before, api.predict(states[-1], data['causal'])):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        api.predict(states[-1], dict(data['causal'], future_endpoint=np.zeros((4, 2))))


def test_failed_screen_keeps_held_readout_closed(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_moving_paired_decision as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(complete_heads=9, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*args, **kwargs):
        raise AssertionError('No additional readout when original fitting screen fails')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw: None, lambda: None)
    a = json.loads((tmp_path/'analysis.json').read_text())
    assert a['status'] == 'fitting_complete_held_readout_not_run'
    assert a['original_rules_unchanged'] and not a['identity_certification']
