from copy import deepcopy

import pytest
import torch

from test_m3w_context_conditioning import inputs
from test_m3w_supervised_intervention import datasets, training_config, register
from src.world_model.m3w_supervised_intervention import (
    build_forecaster, collate_forecasts, load_verified_forecaster, train_forecaster,
)
from src.world_model.m3w_baseline_relative_forecaster import radial_squash, past_motion_budget


def architecture(mode):
    return {'width': 8, 'heads': 2, 'layers': 1, 'neighbor_policy': 'complete_aligned_history',
            'input_conditioning': 'observed_joint_max_norm', 'output_parameterization': mode}


@pytest.mark.parametrize('mode', ['baseline_skip', 'motion_bounded'])
def test_zero_initialized_residual_is_exact_cv_without_extra_parameters(mode):
    torch.manual_seed(17)
    plain = build_forecaster({k: v for k, v in architecture(mode).items() if k != 'output_parameterization'})
    torch.manual_seed(17)
    model = build_forecaster(architecture(mode))
    x = inputs()
    torch.testing.assert_close(model(x), x['baseline'], atol=0, rtol=0)
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in model.parameters())
    for name, value in plain.state_dict().items():
        if '.output.2.' not in name:
            torch.testing.assert_close(value, model.predictor.state_dict()[name], atol=0, rtol=0)


def test_radial_squash_has_gradient_at_zero_and_stable_large_vectors():
    x = torch.tensor([[0., 0.], [1e30, -1e30]], requires_grad=True)
    y = radial_squash(x)
    assert torch.isfinite(y).all()
    assert (y.norm(dim=-1) <= 1.000001).all()
    y[0].sum().backward()
    torch.testing.assert_close(x.grad[0], torch.ones(2))
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    torch.testing.assert_close(radial_squash(x.detach() @ rotation), y.detach() @ rotation)


def test_bound_uses_only_ego_motion_and_requested_time_not_neighbor_distance():
    x = inputs()
    before = past_motion_budget(x)
    torch.testing.assert_close(before, torch.tensor([[.25, .5]]))
    x['neighbors'][..., :2] *= 100
    torch.testing.assert_close(past_motion_budget(x), before)
    model = build_forecaster(architecture('motion_bounded'))
    with torch.no_grad():
        model.predictor.predictor.output[-1].bias.fill_(100.)
    delta = model(x) - x['baseline']
    assert (delta.norm(dim=-1) <= before + 1e-6).all()


def test_exact_stationary_history_stays_floor_without_discarding_requested_labels():
    x = inputs()
    x['history'][..., :2] = 0
    x['baseline'].zero_()
    model = build_forecaster(architecture('motion_bounded'))
    with torch.no_grad():
        model.predictor.predictor.output[-1].bias.fill_(2.)
    torch.testing.assert_close(model(x), x['baseline'], atol=0, rtol=0)
    assert model(x).shape[1] == 2


def test_future_inputs_refused_and_masked_requests_remain_zero():
    x = inputs()
    model = build_forecaster(architecture('motion_bounded'))
    with pytest.raises(ValueError, match='schema'):
        model({**x, 'future_endpoint': torch.ones(1, 2)})
    x['request_mask'][0, 1] = False
    x['baseline'][0, 1] = float('nan')
    x['prediction_time'][0, 1] = float('nan')
    torch.testing.assert_close(model(x)[0, 1], torch.zeros(2))


def test_new_parameterization_refuses_unknown_or_public_core_modes():
    with pytest.raises(ValueError, match='parameterization'):
        build_forecaster(architecture('future_scale'))
    with pytest.raises(ValueError, match='Transformer'):
        build_forecaster({**architecture('motion_bounded'), 'family': 'eqmotion_fixed_head'})


def test_real_training_resume_and_verified_reload_match(tmp_path):
    torch.set_num_threads(2)
    contract, train, _ = datasets(tmp_path)
    _, settings = training_config()
    settings['objective'] = 'smooth_l1'
    spec = architecture('motion_bounded')
    full = train_forecaster(train, architecture=spec, settings=settings, output_dir=tmp_path/'full')
    train_forecaster(train, architecture=spec, settings=settings, output_dir=tmp_path/'resume', stop_after=3)
    resumed = train_forecaster(train, architecture=spec, settings=settings, output_dir=tmp_path/'resume', resume=True)
    a = torch.load(full['checkpoint'], weights_only=True)
    b = torch.load(resumed['checkpoint'], weights_only=True)
    assert a['losses'] == b['losses']
    assert a['model']['predictor.predictor.output.2.weight'].abs().sum() > 0
    for name, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][name], atol=0, rtol=0)
    verified = load_verified_forecaster(register(contract, full), 'fold_a', device='cpu')
    batch = collate_forecasts([train[0], train[1]])
    pred = verified(batch['inputs'])
    assert torch.isfinite(pred).all()
    changed = deepcopy(batch['inputs'])
    changed['neighbors'][~changed['neighbor_mask']] = float('nan')
    torch.testing.assert_close(verified(changed), pred, atol=0, rtol=0)
