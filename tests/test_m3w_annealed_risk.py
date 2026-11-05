import numpy as np
import pytest
import torch
from src.world_model import m3w_annealed_risk as api
from src.world_model import m3w_signed_decision as old
from tests.test_m3w_admission_priority import prepared


def test_schedule_full_budget_not_resume_segment():
    rates = [api.learning_rate('cosine', .001, i, 2000) for i in range(2000)]
    assert rates[0] == .001 and rates[-1] == .00001
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert api.learning_rate('constant', .001, 1999, 2000) == .001
    with pytest.raises(ValueError):
        api.learning_rate('cosine', .001, 2000, 2000)


def test_real_updates_resume_exact_control_and_future_exclusion(tmp_path):
    torch.set_num_threads(1); d, warm, settings = prepared(tmp_path); states = []
    for arm in api.ARMS:
        kw = dict(arm=arm, settings=settings, identity={'fixed': True})
        path = tmp_path/(arm+'.pt.gz')
        api.fit(d, warm, **kw, path=path, stop_at=2)
        a = api.fit(d, warm, **kw, path=path, resume=True)
        b = api.fit(d, warm, **kw, path=tmp_path/(arm+'_repeat.pt.gz'))
        api.parent.parent.parent.router.exact_state(a, b)
        np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        assert not torch.equal(a['model']['encoder.weight'], warm['model']['encoder.weight'])
        states.append(a)
    api.assert_matched(*states)
    control = old.fit(d, warm, arm='continued_mse', settings=settings, identity={'fixed': True}, path=tmp_path/'old.pt.gz')
    for k in ('model', 'optimizer', 'sampler', 'draw_hash', 'input_hash', 'target_hash', 'proposal_hash'):
        api.base.exact(states[0][k], control[k])
    prediction = api.predict(states[1], d['causal'])
    d['truth_by_arm']['bound_aware'][:] = 12345
    np.testing.assert_array_equal(prediction, api.predict(states[1], d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(states[1], dict(d['causal'], future_endpoint=np.zeros((4, 2))))


def test_resume_schedule_identity_cannot_change(tmp_path):
    torch.set_num_threads(1); d, warm, settings = prepared(tmp_path)
    path = tmp_path/'model.pt.gz'
    api.fit(d, warm, arm='cosine', settings=settings, identity={}, path=path, stop_at=2)
    with pytest.raises((ValueError, AssertionError)):
        api.fit(d, warm, arm='constant', settings=settings, identity={}, path=path, resume=True)


def test_protocol_and_decision_before_outcome_readout():
    import ast
    from pathlib import Path
    from scripts import run_m3w_moving_annealed_risk as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['new_heads'] == 6 and cfg['risk_budget'] == cfg['easy_net_budget'] == .02
    assert not any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    f = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert f['evaluate'].index('api.parent.parent.choose(') < f['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 6" in f['evaluate']
