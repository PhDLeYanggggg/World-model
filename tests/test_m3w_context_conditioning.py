import pytest
import torch
from torch import nn

from src.world_model.m3w_context_conditioning import ObservedContextConditioner, condition_inputs


def inputs():
    return {'history': torch.tensor([[[-.5, 0., -.2], [-.2, 0., -.1], [0., 0., 0.]]]),
        'history_mask': torch.ones(1, 3, dtype=torch.bool),
        'neighbors': torch.tensor([[[[300., 400., -.2], [300., 400., -.1], [300., 400., 0.]],
                                     [[9999., 0., -.2], [9999., 0., -.1], [9999., 0., 0.]]]]),
        'neighbor_mask': torch.tensor([[[True, True, True], [False, True, True]]]),
        'baseline': torch.tensor([[[.2, 0.], [.4, 0.]]]),
        'prediction_time': torch.tensor([[.5, 1.]]), 'request_mask': torch.ones(1, 2, dtype=torch.bool)}


class Echo(nn.Module):
    def forward(self, x):
        return x['baseline']


def test_observed_joint_scale_ignores_ineligible_neighbor_and_future_requests():
    x = inputs()
    normalized, scale = condition_inputs(x)
    assert scale.item() == 500.
    torch.testing.assert_close(normalized['neighbors'][0, 0, :, :2].norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(normalized['history'][..., 2], x['history'][..., 2], rtol=0, atol=0)
    assert torch.equal(normalized['neighbor_mask'], x['neighbor_mask'])
    x['baseline'] *= 1000
    x['prediction_time'] *= 10
    assert condition_inputs(x)[1].item() == 500.


def test_output_restored_and_original_payload_not_modified():
    x = inputs()
    original = {k: v.clone() for k, v in x.items()}
    torch.testing.assert_close(ObservedContextConditioner(Echo())(x), x['baseline'])
    for key in x:
        assert torch.equal(x[key], original[key])


def test_factor_one_is_exact_identity():
    x = inputs()
    x['neighbor_mask'][:] = False
    normalized, scale = condition_inputs(x)
    assert scale.item() == 1.
    for key in ('history', 'baseline'):
        assert torch.equal(normalized[key], x[key])


def test_future_input_or_nonfinite_observed_context_rejected_before_mask_filter():
    x = inputs()
    with pytest.raises(ValueError, match='schema'):
        condition_inputs({**x, 'future_endpoint': torch.ones(1, 2)})
    x['neighbors'][0, 1, 1, 2] = 1.
    with pytest.raises(ValueError, match='past'):
        condition_inputs(x)
    x = inputs()
    x['neighbors'][0, 1, 1, 0] = float('inf')
    with pytest.raises(ValueError, match='[Ff]inite'):
        condition_inputs(x)


def test_rotation_preserves_radial_conditioner():
    x = inputs()
    before = condition_inputs(x)[1]
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    x['history'][..., :2] = x['history'][..., :2] @ rotation
    x['neighbors'][..., :2] = x['neighbors'][..., :2] @ rotation
    torch.testing.assert_close(condition_inputs(x)[1], before)


def test_output_loss_and_gradient_stay_in_original_units():
    class Gain(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def forward(self, x):
            return self.weight * x['baseline']

    plain, wrapped = Gain(), ObservedContextConditioner(Gain())
    x = inputs()
    target = torch.ones_like(x['baseline'])
    a, b = (nn.functional.smooth_l1_loss(m(x), target) for m in (plain, wrapped))
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(plain.weight.grad, wrapped.predictor.weight.grad)


def test_factory_wrapper_adds_no_parameters_and_refuses_unknown_mode():
    from src.world_model.m3w_supervised_intervention import build_forecaster
    spec = {'width': 8, 'heads': 2, 'layers': 1}
    torch.manual_seed(17)
    plain = build_forecaster(spec)
    torch.manual_seed(17)
    wrapped = build_forecaster({**spec, 'input_conditioning': 'observed_joint_max_norm'})
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in wrapped.parameters())
    for name, value in plain.state_dict().items():
        assert torch.equal(value, wrapped.predictor.state_dict()[name])
    assert wrapped.source_identity['output_restored_before_loss_and_evaluation']
    with pytest.raises(ValueError, match='Unknown'):
        build_forecaster({**spec, 'input_conditioning': 'from_targets'})
