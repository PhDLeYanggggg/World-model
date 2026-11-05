import numpy as np
import pytest
import torch
from src.world_model import m3w_admission_priority as api
from tests.test_m3w_kinematic_risk import fixture


def prepared(tmp_path):
    d = fixture(); settings = dict(width=4, learning_rate=.001, query_batch_size=2,
        minimum_updates=4, largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    initial = api.parent.parent.fit(d, arm='mean', seed=17, settings=settings,
        identity={}, path=tmp_path/'original.pt.gz')
    d = api.parent.prepare(d, initial)
    warm = api.parent.fit(d, initial, arm='kinematic', settings=settings,
        identity={}, path=tmp_path/'warm.pt.gz')
    return api.prepare(d, warm), warm, settings


def test_priority_loss_exact_control_gradient_and_unknowns(tmp_path):
    d, warm, _ = prepared(tmp_path); ix = np.arange(4); seg = torch.tensor([0, 0, 1, 1])
    eligible = np.argwhere(d['causal']['allowed'] & d['observed_future'][:, None])[0]
    i, j = map(int, eligible)
    d['truth_by_arm']['bound_aware'][i, j, 4] = 1
    d['truth_by_arm']['bound_aware'][i, j, 3] = 0
    pred = torch.zeros(4, 7, 2, requires_grad=True)
    old = api.parent.parent.objective(pred, d, ix, seg, 2, 'mean')['total']
    control = api.objective(pred, d, ix, seg, 2, 'continued_mse')['total']
    assert torch.equal(old, control)
    g0 = torch.autograd.grad(control, pred, retain_graph=True)[0]
    d['priority_mask'][:] = False
    d['priority_mask'][i, j, 1] = True
    treated = api.objective(pred, d, ix, seg, 2, 'false_safe_priority')['total']
    g1 = torch.autograd.grad(treated, pred)[0]
    assert g0[i, j, 1].abs() > 0
    torch.testing.assert_close(g1[i, j, 1], 10*g0[i, j, 1])
    d['priority_mask'][:] = True; d['observed_future'][:] = False
    with pytest.raises(ValueError, match='Unknown'):
        api.objective(pred, d, ix, seg, 2, 'false_safe_priority')


def test_actual_updates_resume_and_future_not_input(tmp_path):
    torch.set_num_threads(1); d, warm, settings = prepared(tmp_path)
    before = api.parent.predict(warm, d['causal']); states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(arm=arm, settings=settings, identity={'fixed': True})
        initial = api.initialize(warm)
        for key in warm['model']:
            assert torch.equal(initial.state_dict()[key], warm['model'][key])
        api.fit(d, warm, **kw, path=path, stop_at=2)
        a = api.fit(d, warm, **kw, path=path, resume=True)
        b = api.fit(d, warm, **kw, path=tmp_path/(arm+'_repeat.pt.gz'))
        api.parent.parent.parent.router.exact_state(a, b)
        np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        states.append(a)
    api.assert_matched(*states)
    prediction = api.predict(states[1], d['causal'])
    assert not np.array_equal(before, prediction)
    d['truth_by_arm']['bound_aware'][:] = 12345; d['priority_mask'][:] = True
    np.testing.assert_array_equal(prediction, api.predict(states[1], d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(states[1], dict(d['causal'], future_endpoint=np.zeros((4, 2))))


def test_registered_rules_freeze_and_decision_before_outcomes():
    import ast
    from pathlib import Path
    from scripts import run_m3w_moving_admission_priority as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['priority_multiplier'] == 10 and cfg['new_heads'] == 6 and cfg['primary_updates'] == 12000
    assert cfg['risk_budget'] == cfg['easy_net_budget'] == .02
    assert not any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    f = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert f['evaluate'].index('api.parent.parent.choose(') < f['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 6" in f['evaluate']
