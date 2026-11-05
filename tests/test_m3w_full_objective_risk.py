import numpy as np
import pytest
import torch
from src.world_model import m3w_full_objective_risk as api
from tests.test_m3w_admission_priority import prepared


SETTINGS = dict(outer_steps=6, history_size=4, tolerance_grad=1e-7,
    tolerance_change=1e-12, checkpoint_every=1, heartbeat_every=1)


def test_complete_gradient_matches_unchunked_objective(tmp_path):
    torch.set_num_threads(1); d, origin, _ = prepared(tmp_path)
    p = api.prepare(d, origin); model = api.model_from(origin)
    result = api.full(model, p, True)
    gradients = [v.grad.clone() for v in model.parameters()]
    model.zero_grad(set_to_none=True)
    pred = model(p['z'].double(), torch.tensor([0, 0, 1, 1]), 2)
    loss = api.task_losses(pred, p['target'], p['allowed'], p['weights']).mean()
    loss.backward()
    assert abs(float(loss.detach())-result['mean_MSE']) < 1e-12
    for parameter, expected in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-10, atol=1e-12)


def test_actual_training_exact_resume_and_causal_only_inference(tmp_path):
    torch.set_num_threads(1); d, origin, _ = prepared(tmp_path)
    kw = dict(settings=SETTINGS, identity={'fixed': True})
    path = tmp_path/'resumed.pt.gz'
    api.fit(d, origin, **kw, path=path, stop_at=3)
    a = api.fit(d, origin, **kw, path=path, resume=True)
    b = api.fit(d, origin, **kw, path=tmp_path/'direct.pt.gz')
    for k in ('model', 'optimizer', 'step', 'trace', 'full_gradient_passes', 'changed_parameter_steps', 'final_fitting'):
        api.base.exact(a[k], b[k])
    assert a['changed_parameter_steps'] > 0
    assert a['final_fitting']['mean_MSE'] < a['initial_fitting']['mean_MSE']
    assert not torch.equal(a['model']['encoder.weight'], a['initial_model']['encoder.weight'])
    prediction = api.predict(a, d['causal'])
    d['truth_by_arm']['bound_aware'][:] = 1234
    np.testing.assert_array_equal(prediction, api.predict(a, d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(a, dict(d['causal'], future_endpoint=np.ones((4, 2))))


def test_interruption_does_not_save_line_search_trial(tmp_path):
    torch.set_num_threads(1); d, origin, _ = prepared(tmp_path)
    path = tmp_path/'state.pt.gz'; kw = dict(settings=SETTINGS, identity={})
    saved = api.fit(d, origin, **kw, path=path, stop_at=1)
    original_bytes = path.read_bytes(); calls = 0
    def interrupt():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise KeyboardInterrupt('inside line search')
    with pytest.raises(KeyboardInterrupt):
        api.fit(d, origin, **kw, path=path, resume=True, guard=interrupt)
    assert path.read_bytes() == original_bytes
    a = api.fit(d, origin, **kw, path=path, resume=True)
    b = api.fit(d, origin, **kw, path=tmp_path/'reference.pt.gz')
    for k in ('model', 'optimizer', 'trace', 'full_gradient_passes'):
        api.base.exact(a[k], b[k])
    assert a['step'] > saved['step']


def test_lower_MSE_alone_does_not_open_held_readout():
    rows = [dict(job=str(i), initial_fitting=dict(mean_MSE=1., observed_sign_regret=[1., 1.]),
        final_fitting=dict(mean_MSE=.5, observed_sign_regret=[.5, 1.1])) for i in range(3)]
    assert not api.fitting_screen(rows)['followup_readout_allowed']
    for r in rows:
        r['final_fitting']['observed_sign_regret'] = [.5, .5]
    result = api.fitting_screen(rows)
    assert result['followup_readout_allowed'] and not result['safety_certified']
