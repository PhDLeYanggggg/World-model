import numpy as np
import pytest
import torch

from src.world_model import m3w_signed_decision as api
from tests.test_m3w_admission_priority import prepared


def test_signed_cost_boundary_not_majority_class():
    # Nine small positive costs and one larger negative cost: event majority is unsafe,
    # but mean signed cost is negative. The cost-aware optimum must stay negative.
    y = torch.tensor([.1]*9+[-2.], dtype=torch.float64)
    optimum = np.log(.9/2.)
    score = torch.full_like(y, optimum, requires_grad=True)
    grad = torch.autograd.grad(api.signed_cost(score, y).mean(), score)[0].sum()
    assert abs(float(grad)) < 1e-12 and optimum < 0
    s = torch.zeros_like(y, requires_grad=True)
    g = torch.autograd.grad(api.signed_cost(s, y).sum(), s)[0]
    torch.testing.assert_close(g, -2*y, rtol=0, atol=0)
    assert api.signed_cost(torch.tensor([1e5, -1e5]), torch.tensor([-1., 1.])).isfinite().all()
    with pytest.raises(ValueError, match='detached'):
        api.signed_cost(y, y.clone().requires_grad_())


def test_control_exact_unknown_bounds_and_empty_queries(tmp_path):
    d, _, _ = prepared(tmp_path); ix = np.arange(4); seg = torch.tensor([0, 0, 1, 1])
    d['observed_future'][:] = False
    pred = torch.randn(4, 7, 2, requires_grad=True)
    old = api.parent.parent.objective(pred, d, ix, seg, 2, 'mean')['total']
    control = api.objective(pred, d, ix, seg, 2, 'continued_mse')['total']
    treatment = api.objective(pred, d, ix, seg, 2, 'signed_cost_logistic')['total']
    assert torch.equal(old, control) and torch.equal(old, treatment)
    d['causal']['allowed'][:] = False
    zero = api.objective(pred, d, ix, seg, 2, 'signed_cost_logistic')['total']
    assert float(zero.detach()) == 0 and torch.equal(torch.autograd.grad(zero, pred)[0], torch.zeros_like(pred))


def test_actual_training_resume_matching_and_future_exclusion(tmp_path):
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
    p = api.predict(states[1], d['causal'])
    d['truth_by_arm']['bound_aware'][:] = 12345
    np.testing.assert_array_equal(p, api.predict(states[1], d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(states[1], dict(d['causal'], future_endpoint=np.zeros((4, 2))))


def test_protocol_and_inference_before_outcomes():
    import ast
    from pathlib import Path
    from scripts import run_m3w_moving_signed_decision as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['new_heads'] == 6 and cfg['risk_budget'] == cfg['easy_net_budget'] == .02
    assert not any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    f = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert f['evaluate'].index('api.parent.parent.choose(') < f['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 6" in f['evaluate']
