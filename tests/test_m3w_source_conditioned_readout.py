import numpy as np
import pytest
import torch

from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics
from src.world_model.m3w_source_temporal_centered import CenteredTemporalDynamics
from src.world_model.m3w_source_conditioned_readout import ConditionedTemporalDynamics, training_readout_gain


@pytest.mark.parametrize('arm', ['geometry', 'centered'])
def test_unit_gain_matches_old_forward_and_initialization(arm):
    torch.manual_seed(17)
    old = TemporalSourceDynamics('geometry') if arm == 'geometry' else CenteredTemporalDynamics('centered')
    torch.manual_seed(17)
    new = ConditionedTemporalDynamics(arm, 1.)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    with torch.no_grad():
        old.head[-1].weight.normal_(std=.01)
        new.load_state_dict(dict(old.state_dict(), readout_gain=new.readout_gain))
    inputs = (torch.randn(3, 480), torch.randn(3, 8, 512), torch.ones(3, 8))
    torch.testing.assert_close(new(*inputs), old(*inputs), rtol=0, atol=0)
    assert sum(v.numel() for v in new.parameters()) == 63960


@pytest.mark.parametrize('arm', ['geometry', 'centered'])
def test_constant_output_reparameterization_preserves_function_class_and_bounds(arm):
    old = TemporalSourceDynamics('geometry') if arm == 'geometry' else CenteredTemporalDynamics('centered')
    new = ConditionedTemporalDynamics(arm, 530.)
    with torch.no_grad():
        old.head[-1].weight.normal_(std=.3)
        old.head[-1].bias.normal_(std=.3)
        new.load_state_dict(dict(old.state_dict(), readout_gain=new.readout_gain))
        new.head[-1].weight.mul_(530)
        new.head[-1].bias.mul_(530)
    inputs = (torch.randn(8, 480), torch.randn(8, 8, 512), torch.ones(8, 8))
    torch.testing.assert_close(new(*inputs), old(*inputs), rtol=1e-5, atol=1e-6)
    assert torch.all(torch.linalg.vector_norm(new(*inputs), dim=-1) < 1)


def test_gain_is_fixed_training_median_and_rejects_missing_support():
    assert training_readout_gain([0, 20, 40], .5) == 40
    assert training_readout_gain([.1, .2], 1) == 1
    for radii, scale in [([0, 0], 1), ([np.nan, 1], 1), ([1, 2], 0), ([-1, 2], 1)]:
        with pytest.raises(ValueError):
            training_readout_gain(radii, scale)


def test_initial_readout_jacobian_is_scaled_not_zeroed():
    old, new = ConditionedTemporalDynamics('geometry', 1.), ConditionedTemporalDynamics('geometry', 500.)
    new.load_state_dict(dict(old.state_dict(), readout_gain=new.readout_gain))
    inputs = (torch.randn(3, 480), torch.randn(3, 8, 512), torch.ones(3, 8))
    old(*inputs).sum().backward(); new(*inputs).sum().backward()
    torch.testing.assert_close(new.head[-1].bias.grad*500, old.head[-1].bias.grad)
