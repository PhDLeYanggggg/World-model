import numpy as np
import pytest
import torch
from src.world_model import m3w_decoupled_risk as api
from tests.test_m3w_admission_priority import prepared as prior_prepared


def prepared(tmp_path):
    d, warm, settings = prior_prepared(tmp_path)
    warm = dict(warm, observed_mask_hash=api.kin.digest(d['observed_future']))
    return d, warm, dict(settings, learning_rate=.0001)


def test_initial_function_and_capacity_controls(tmp_path):
    torch.set_num_threads(1); d, warm, _ = prepared(tmp_path)
    x = api.kin.batch_features(d, 'kinematic'); seg = torch.tensor([0, 0, 1, 1])
    original = api.kin.KinematicRiskHead(4, 17); original.load_state_dict(warm['model'])
    expected = original(x, seg, 2)
    for arm in api.ARMS:
        model = api.initialize(warm, arm)
        r, s = model(x, seg, 2)
        torch.testing.assert_close(r, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(s, expected, atol=1e-6, rtol=1e-6)
    narrow, wide, split = [api.Model(a, 64, 17) for a in api.ARMS]
    assert [sum(p.numel() for p in m.parameters()) for m in (narrow, wide, split)] == [32260, 64516, 64260]


def test_decision_gradient_is_separate_only_in_split_arm(tmp_path):
    d, warm, _ = prepared(tmp_path); x = api.kin.batch_features(d, 'kinematic')
    seg = torch.tensor([0, 0, 1, 1])
    for arm in api.ARMS:
        model = api.initialize(warm, arm); r, s = model(x, seg, 2)
        s.sum().backward()
        if arm == 'separate64':
            assert all(p.grad is None for p in model.risk.parameters())
            assert model.decision.encoder.weight.grad.abs().sum() > 0
        else:
            assert model.shared.encoder.weight.grad.abs().sum() > 0


def test_original_bound_supervision_and_sign_conjunction(tmp_path):
    d, warm, _ = prepared(tmp_path); ix = np.arange(4); seg = torch.tensor([0, 0, 1, 1])
    d['observed_future'][np.flatnonzero(d['causal']['allowed'].any(1))[0]] = False
    r = torch.zeros(4, 7, 2, requires_grad=True); s = torch.zeros_like(r, requires_grad=True)
    out = api.objective((r, s), d, ix, seg, 2)
    reference = api.kin.parent.objective(r, d, ix, seg, 2, 'mean')['total']
    assert torch.equal(out['risk_MSE'], reference)
    out['total'].backward()
    unknown = (~d['observed_future'])[:, None] & d['causal']['allowed']
    assert unknown.any() and r.grad[unknown].abs().sum() > 0 and s.grad[unknown].abs().sum() > 0
    r = np.array([[-1., 2.], [-1., -2.]])
    s = np.array([[3., -1.], [-3., 4.]])
    assert np.array_equal(api.combined(r, s) <= 0, (r <= 0) & (s <= 0))


def test_actual_training_resume_causal_prediction_and_matching(tmp_path):
    torch.set_num_threads(1); d, warm, settings = prepared(tmp_path); states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(settings=settings, identity={}, arm=arm)
        api.fit(d, warm, **kw, path=path, stop_at=2)
        a = api.fit(d, warm, **kw, path=path, resume=True)
        b = api.fit(d, warm, **kw, path=tmp_path/(arm+'_repeat.pt.gz'))
        api.exact_state(a, b)
        for left, right in zip(api.predict(a, d['causal']), api.predict(b, d['causal'])):
            np.testing.assert_array_equal(left, right)
        assert any(not torch.equal(v, a['initial_model'][k]) for k, v in a['model'].items())
        states.append(a)
    api.assert_matched(states)
    before = api.predict(states[-1], d['causal'])
    d['truth_by_arm']['bound_aware'][:] = 1e6; d['observed_future'][:] = False
    for a, b in zip(before, api.predict(states[-1], d['causal'])):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='causal'):
        api.predict(states[-1], dict(d['causal'], future_endpoint=np.zeros((4, 2))))


def test_full_diagnostic_uses_complete_query_weighting(tmp_path):
    d, warm, _ = prepared(tmp_path); m = api.initialize(warm, 'separate64')
    a = api.full(m, d, warm['sampler']['counts'])
    assert np.isfinite(a['mean_MSE']) and a['mean_MSE'] > 0
    assert a['observed_sign_regret'] == a['risk_only_sign_regret'] == a['decision_only_sign_regret']
