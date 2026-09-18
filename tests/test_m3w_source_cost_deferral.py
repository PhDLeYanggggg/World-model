import copy
import hashlib

import numpy as np
import pytest
import torch

from src.world_model.m3w_source_cost_dynamics import SourceDynamics, fit_dynamics, restore, forecast
from src.world_model.m3w_source_modality_continuation import continue_modality
from src.world_model.m3w_source_cost_deferral import CostDeferral, deferral_loss, hard_action, fit_deferral, predict_deferral


def fixture(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    x = (torch.randn(8, 6), torch.rand(8, 8, 3, 32, 32), torch.ones(8, 8, 1, 32, 32))
    frame = (torch.ones(8), torch.eye(2)[None].repeat(8, 1, 1), torch.ones(8, dtype=torch.bool))
    y = torch.randn(8, 12, 2)*.1
    y[:4] = 0
    ids, weights = np.arange(8), np.ones(8)/8
    def inputs(i):
        return tuple(v[i] for v in x), tuple(v[i] for v in frame)
    config = dict(updates=3, batch_size=4, learning_rate=.0003, weight_decay=.0001, checkpoint_every=2)
    model = SourceDynamics(6)
    fit_dynamics(model, inputs, lambda i:y[i], ids, weights, arm='mask_only', objective='ade', normalizer=1.,
        seed=17, config=config, identity={'parent':True}, checkpoint=tmp_path/'parent.pt', heartbeat=lambda **_:None)
    parent = torch.load(tmp_path/'parent.pt', weights_only=False)
    config = dict(config, start_step=3, updates=10, minimum_lr_ratio=.01, milestones=[3, 6, 10])
    return inputs, y, ids, weights, parent, config


def run(tmp_path, payload, name, variant, stop=None):
    inputs, y, ids, weights, parent, config = payload
    model = CostDeferral(6)
    result = fit_deferral(model, inputs, lambda i:y[i], ids, weights, parent=parent, variant=variant,
        config=config, identity={'variant':variant}, checkpoint=tmp_path/name, heartbeat=lambda **_:None,
        snapshot=lambda _:None, stop_at=stop)
    return model, result


def test_proposal_matches_parent_forward_and_tie_is_exact_baseline(tmp_path):
    payload = fixture(tmp_path)
    model, _ = run(tmp_path, payload, 'initial.pt', 'expected_cost', stop=3)
    old = SourceDynamics(6)
    old.load_state_dict(payload[4]['model'])
    p, z = predict_deferral(model, payload[0], payload[2])
    np.testing.assert_array_equal(p, forecast(old, payload[0], payload[2], 'mask_only'))
    assert not z.any()
    assert not hard_action(torch.from_numpy(p), torch.from_numpy(z)).any()


def test_dense_control_exactly_matches_registered_continuation(tmp_path):
    payload = fixture(tmp_path)
    inputs, y, ids, weights, parent, config = payload
    original = copy.deepcopy(parent['optimizer'])
    extended, _ = run(tmp_path, payload, 'extended.pt', 'dense_control')
    old = SourceDynamics(6)
    continue_modality(old, inputs, lambda i:y[i], ids, weights, parent=parent, arm='mask_only', schedule='cosine',
        config=config, identity={'control':True}, checkpoint=tmp_path/'old.pt', heartbeat=lambda **_:None, snapshot=lambda _:None)
    for name, value in old.state_dict().items():
        assert torch.equal(value, extended.state_dict()[name])
    for k, state in original['state'].items():
        for name in state:
            assert torch.equal(state[name], parent['optimizer']['state'][k][name])


@pytest.mark.parametrize('variant', ['expected_cost', 'cost_supervised'])
def test_interrupted_resume_exact_and_complete_readonly(tmp_path, variant):
    payload = fixture(tmp_path)
    full, _ = run(tmp_path, payload, 'full.pt', variant)
    run(tmp_path, payload, 'resume.pt', variant, stop=7)
    resumed, result = run(tmp_path, payload, 'resume.pt', variant)
    assert result['new_updates'] == 3
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name])
    digest = hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()
    assert run(tmp_path, payload, 'resume.pt', variant)[1]['new_updates'] == 0
    assert digest == hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()
    a, b = [torch.load(tmp_path/p, weights_only=False) for p in ('full.pt','resume.pt')]
    np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])


def test_expected_cost_gradient_rewards_gain_and_discourages_harm():
    target = torch.zeros(2, 12, 2)
    target[1, :, 0] = 1
    proposal = torch.ones(2, 12, 2)
    proposal[:, :, 1] = 0
    score = torch.zeros(2, requires_grad=True)
    loss, _ = deferral_loss(proposal, score, target, 1., 'expected_cost')
    loss.backward()
    assert score.grad[0] > 0 and score.grad[1] < 0
    torch.testing.assert_close(hard_action(proposal, torch.tensor([-1.,1.])), target, rtol=0, atol=0)


def test_cost_label_has_no_backpropagation_into_target_proposal():
    proposal = torch.ones(2, 12, 2, requires_grad=True)
    score = torch.tensor([.2, -.3], requires_grad=True)
    target = torch.zeros_like(proposal)
    total, terms = deferral_loss(proposal, score, target, 1., 'cost_supervised')
    expected, _ = deferral_loss(proposal, score, target, 1., 'expected_cost')
    ade = torch.linalg.vector_norm(proposal-target, dim=-1).mean()
    assert torch.autograd.grad(terms['cost_fit'], proposal, allow_unused=True, retain_graph=True)[0] is None
    grad = torch.autograd.grad(total, proposal, retain_graph=True)[0]
    reference = torch.autograd.grad(expected+.1*ade, proposal)[0]
    torch.testing.assert_close(grad, reference, rtol=2e-7, atol=1e-9)


def test_gate_input_is_observed_only_and_mask_arm_ignores_rgb():
    torch.manual_seed(4)
    model = CostDeferral(6)
    with torch.no_grad():
        model.risk_gate.weight.fill_(.01)
    x, rgb, coverage = torch.randn(2,6), torch.rand(2,8,3,32,32), torch.ones(2,8,1,32,32)
    first = model.proposal_and_score(x, rgb, coverage, 'mask_only')
    second = model.proposal_and_score(x, rgb*100+5, coverage, 'mask_only')
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        deferral_loss(first[0], first[1], first[0], 0., 'expected_cost')
