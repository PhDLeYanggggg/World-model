"""Direct two-actor admission with an additive matched control."""

import torch
from torch import nn

from src.world_model.m3w_reference_bridge_policy import WIDTH


class PairAdmission(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU())
        self.unary = nn.Linear(24, 2)
        self.interaction = nn.Sequential(nn.Linear(56, 40), nn.SiLU(), nn.Linear(40, 2))

    def forward(self, actions, geometry):
        hidden = self.encode(actions)
        x, y = hidden.unbind(1)
        additive = self.unary(hidden).sum(1)
        residual = self.interaction(torch.cat((x+y, (x-y).abs(), geometry), dim=1))
        return torch.stack((additive, additive+residual), dim=1)
