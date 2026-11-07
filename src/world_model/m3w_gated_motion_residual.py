"""Deterministic OLS4-relative velocity corrections learned from past motion."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the native arm64 .venv-pytorch environment')

import torch
from torch import nn
from torch.nn.functional import interpolate


class GatedMotionResidual(nn.Module):
    def __init__(self, gated):
        super().__init__()
        self.gated = gated
        self.decode = nn.Sequential(nn.Linear(16, 48), nn.SiLU(), nn.Linear(48, 48),
                                    nn.SiLU(), nn.Linear(48, 8))
        self.register_buffer('ols_weights', torch.tensor([-.3, -.1, .1, .3]))

    def forward(self, inputs):
        history = inputs['history'][..., :2]
        # Center before the slope calculation to retain translation invariance.
        centered = history-history[:, -1:]
        velocity = (centered[:, -4:] * self.ols_weights[None, :, None]).sum(1)
        delta = centered.diff(dim=1)
        scale = delta.square().sum(2).mean(1).sqrt().clamp_min(.05)
        dominant = delta[torch.arange(len(delta), device=delta.device), delta.square().sum(2).argmax(1)]
        axis = torch.where((velocity.square().sum(1) > 1e-10)[:, None], velocity, dominant)
        axis = axis / torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
        perp = torch.stack((-axis[:, 1], axis[:, 0]), 1)
        frame = torch.stack((axis, perp), dim=2)
        local = (delta @ frame)/scale[:, None, None]
        feat = torch.cat((local.flatten(1), torch.log1p(scale)[:, None],
                          (torch.linalg.vector_norm(velocity, dim=1)/scale)[:, None]), dim=1)
        output = self.decode(feat)
        knots = output[:, :6].reshape(-1, 2, 3).tanh()
        bounds = knots.new_tensor([2., 1.])[None, :, None]
        gate = output[:, 6:].sigmoid()[..., None] if self.gated else 1.
        increments = interpolate(knots*bounds*gate, size=12, mode='linear', align_corners=True).transpose(1, 2)
        world = (increments @ frame.transpose(1, 2))*scale[:, None, None]
        return (world+velocity[:, None]).cumsum(1)
