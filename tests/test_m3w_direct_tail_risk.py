import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from src.world_model import m3w_direct_tail_risk as api
from tests.test_m3w_proposal_conditioned_risk import learning


def data():
    d = learning()
    logits = np.tile(np.arange(7), (4, 1)).astype(float)
    return api.parent.condition(d, logits)[0]


def test_direct_targets_unknown_bounds_and_quantile_gradient():
    d = data(); d['observed_future'][0] = False
    t = d['truth_by_arm']['bound_aware']
    t[0] = 0; t[0, :, 1] = 1; t[0, :, 4] = 1
    assert torch.equal(api.signed(t[0, 6]), torch.ones(2))
    p = torch.full((4, 7, 2), -1000., requires_grad=True)
    terms = api.objective(p, d, np.arange(4), torch.tensor([0, 0, 1, 1]), 2, 'quantile90')
    assert terms['unknown_bound_supervised_rows'] == 1
    terms['total'].backward()
    assert (p.grad[:, 6] < 0).all() and not p.grad[:, :6].any()
    # Unbounded linear outputs avoid sigmoid/softmax saturation of the risk gradient.
    p2 = torch.full((4, 7, 2), -1e6, requires_grad=True)
    api.objective(p2, d, np.arange(4), torch.tensor([0, 0, 1, 1]), 2, 'quantile90')['total'].backward()
    assert torch.equal(p.grad, p2.grad)


def test_budget_and_empty_support_not_certified():
    d = data(); c = d['causal']; logits = np.tile(np.arange(7), (4, 1)).astype(float)
    p = np.zeros((4, 7, 5)); p[..., 2:4] = 1
    e = np.full((4, 7, 2), -.1); e[0, 6, 0] = 1e-9; e[1, 6, 1] = 1e-9; p[2, 6, 3] = 0
    choice, matched = api.choose(logits, p, e, c)
    assert choice.tolist() == [-1, -1, -1, 6]
    for group in c['groups']:
        np.testing.assert_array_equal(np.sort(choice[group]), np.sort(matched[group]))
    assert sum(v.numel() for v in api.DirectRiskHead(428, 64, 17).parameters()) == 28034


def test_exact_resume_matched_loss_and_future_input_rejection(tmp_path):
    torch.set_num_threads(1)
    d = data(); settings = dict(width=4, learning_rate=.001, query_batch_size=2,
        minimum_updates=3, largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    args = dict(seed=17, settings=settings, identity={'fixed': True})
    states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz')
        api.fit(d, **args, arm=arm, path=path, stop_at=1)
        a = api.fit(d, **args, arm=arm, path=path, resume=True)
        b = api.fit(d, **args, arm=arm, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.router.exact_state(a, b)
        np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        states.append(a)
    api.assert_matched(*states)
    c = dict(d['causal'], x=d['causal']['x'].copy()); c['x'][..., 380:] = 999
    np.testing.assert_array_equal(api.predict(a, c), api.predict(a, d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(a, dict(c, future_endpoint=np.zeros((4, 2))))
    d['truth_by_arm']['bound_aware'][:] = 9999
    np.testing.assert_array_equal(api.predict(a, c), api.predict(a, d['causal']))


def test_context_does_not_cross_recording_queries():
    model = api.DirectRiskHead(428, 4, 17); torch.nn.init.normal_(model.excess.weight)
    x = torch.zeros(4, 7, 428); seg = torch.tensor([0, 0, 1, 1])
    a = model(x, seg, 2); x[2:] = 999
    assert torch.equal(a[:2], model(x, seg, 2)[:2])


def test_registered_pilot_and_decision_before_label():
    from scripts import run_m3w_moving_direct_tail_risk as run
    spec = run.source.read(run.CONFIG)
    assert spec['new_heads'] == 6 and spec['primary_updates'] == 12000
    assert spec['risk_budget'] == spec['easy_net_budget'] == .02
    assert not any(spec[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    f = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert f['evaluate'].index('decisions(') < f['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 6" in f['evaluate']
