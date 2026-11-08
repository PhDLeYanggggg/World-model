import torch

from src.world_model import m3w_local_velocity as original
from src.world_model.m3w_velocity_objectives import objective


def batch():
    floor = torch.zeros(2, 12, 2)
    forecast = torch.ones_like(floor, requires_grad=True)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[1] = False
    target = torch.zeros_like(floor)
    target[~valid] = torch.nan
    return forecast, floor, target, valid, torch.ones(2)


def test_preserved_reference_and_factor_isolation():
    p, floor, y, v, scale = batch()
    d = original.lower(p, floor, y, v) / (12*scale)
    torch.testing.assert_close(objective(p, floor, y, v, scale, "grid_harm"), (d+d.relu()).mean())
    plain = objective(p, floor, y, v, scale, "grid_plain")
    masked = objective(p, floor, y, v, scale, "masked_plain")
    torch.testing.assert_close(masked*2, plain)
    torch.testing.assert_close(objective(p, floor, y, v, scale, "masked_harm"), plain)


def test_masked_target_has_no_gradient_for_unknown_steps():
    p, floor, y, v, scale = batch()
    loss = objective(p, floor, y, v, scale, "masked_plain")
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[v].abs().min() > 0
    assert p.grad[~v].abs().max() == 0
