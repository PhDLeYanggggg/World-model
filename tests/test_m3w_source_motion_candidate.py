import hashlib

import numpy as np
import pytest
import torch

from src.world_model.m3w_source_cost_dynamics import SourceDynamics, fit_dynamics, forecast
from src.world_model.m3w_source_modality_continuation import continue_modality
from src.world_model.m3w_source_motion_candidate import motion_objective, fit_motion_candidate


def test_only_zero_target_gradient_is_removed_without_positive_reweighting():
    p = torch.ones(2, 12, 2, requires_grad=True)
    target = torch.zeros_like(p); target[1] = 2
    all_loss, _, _ = motion_objective(p, target, 2., False)
    all_grad = torch.autograd.grad(all_loss, p, retain_graph=True)[0]
    loss, full, count = motion_objective(p, target, 2., True)
    grad = torch.autograd.grad(loss, p)[0]
    assert count == 1 and loss == full / 2
    assert not grad[0].any()
    assert torch.equal(grad[1], all_grad[1])
    assert all_grad[0].any()


def test_all_zero_batch_retains_graph_and_invalid_targets_rejected():
    p = torch.ones(3, 12, 2, requires_grad=True)
    loss, _, count = motion_objective(p, torch.zeros_like(p), 1., True)
    loss.backward()
    assert count == 0 and loss == 0 and not p.grad.any()
    with pytest.raises(ValueError):
        motion_objective(p, torch.full_like(p, float('nan')), 1., True)


def fixture():
    torch.set_num_threads(2); torch.manual_seed(1)
    features = (torch.randn(8, 6), torch.rand(8, 8, 3, 32, 32), torch.ones(8, 8, 1, 32, 32))
    frame = (torch.ones(8), torch.eye(2)[None].repeat(8, 1, 1), torch.ones(8, dtype=torch.bool))
    target = torch.randn(8, 12, 2) * .1; target[:4] = 0
    def inputs(i):
        return tuple(a[i] for a in features), tuple(a[i] for a in frame)
    cfg = dict(start_step=3, updates=10, batch_size=4, learning_rate=.0003,
               weight_decay=.0001, checkpoint_every=2, minimum_lr_ratio=.01)
    return inputs, target, np.arange(8), np.ones(8)/8, cfg


def train(tmp_path, name, payload, suppress=True, stop=None):
    inputs, target, ids, weights, cfg = payload
    torch.manual_seed(17); model = SourceDynamics(6)
    result = fit_motion_candidate(model, inputs, lambda i:target[i], ids, weights, scale=1., seed=17,
        config=cfg, identity={'test':1}, checkpoint=tmp_path/name, heartbeat=lambda **_:None,
        suppress_zero_targets=suppress, stop_at=stop)
    return model, result


def test_control_engine_exactly_matches_registered_two_phase_engine(tmp_path):
    payload = fixture(); inputs, target, ids, weights, cfg = payload
    torch.manual_seed(17); old = SourceDynamics(6)
    first = {k:cfg[k] for k in ['batch_size','learning_rate','weight_decay','checkpoint_every']}
    first['updates'] = cfg['start_step']
    fit_dynamics(old, inputs, lambda i:target[i], ids, weights, arm='mask_only', objective='ade', normalizer=1.,
        seed=17, config=first, identity={'old':1}, checkpoint=tmp_path/'parent.pt', heartbeat=lambda **_:None)
    parent = torch.load(tmp_path/'parent.pt', weights_only=False)
    continue_modality(old, inputs, lambda i:target[i], ids, weights, parent=parent, arm='mask_only', schedule='cosine',
        config=dict(cfg, milestones=[3,10]), identity={'old':1}, checkpoint=tmp_path/'old.pt',
        heartbeat=lambda **_:None, snapshot=lambda _:None)
    new, _ = train(tmp_path, 'new.pt', payload, suppress=False)
    np.testing.assert_array_equal(forecast(old, inputs, ids, 'mask_only'), forecast(new, inputs, ids, 'mask_only'))
    old_state = torch.load(tmp_path/'old.pt', weights_only=False)
    new_state = torch.load(tmp_path/'new.pt', weights_only=False)
    np.testing.assert_array_equal(old_state['draw_counts'], new_state['draw_counts'])
    assert torch.equal(old_state['sampler_rng'], new_state['sampler_rng'])


def test_motion_resume_readonly_and_sampler_matches_control(tmp_path):
    payload = fixture()
    full, _ = train(tmp_path, 'full.pt', payload)
    train(tmp_path, 'resume.pt', payload, stop=5)
    resumed, result = train(tmp_path, 'resume.pt', payload)
    assert result['new_updates'] == 5
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name])
    before = hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()
    assert train(tmp_path, 'resume.pt', payload)[1]['new_updates'] == 0
    assert before == hashlib.sha256((tmp_path/'resume.pt').read_bytes()).hexdigest()
    train(tmp_path, 'control.pt', payload, suppress=False)
    a = torch.load(tmp_path/'full.pt', weights_only=False)
    b = torch.load(tmp_path/'control.pt', weights_only=False)
    np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])
    with pytest.raises(ValueError, match='objective'):
        train(tmp_path, 'full.pt', payload, suppress=False)
