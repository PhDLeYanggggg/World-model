"""Separate easy-reference support from conditional harm and unknown debit."""

import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_reference_bridge_policy as bridge


class FactorValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(bridge.WIDTH, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 8))
        with torch.no_grad():
            self.net[-1].bias[2:] = -4.
            self.net[-1].bias[4] = 0.

    def parts(self, x):
        a = self.net(x)
        return a[..., :2], F.softplus(a[..., 2:4]), a[..., 4], F.softplus(a[..., 5:])

    def forward(self, x):
        gain, base, logit, cond = self.parts(x)
        prob = logit.sigmoid()
        return torch.cat((gain, torch.stack((base[..., 0], prob*cond[..., 0]+cond[..., 2],
                                            base[..., 1], prob*cond[..., 1]), dim=-1)), dim=-1)


def masked_mean(value, mask):
    return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)


def factor_loss(net, x, target, unknown):
    gain, base, logit, cond = net.parts(x)
    support = target[:, 5] > 0.
    l_gain = F.smooth_l1_loss(gain, target[:, :2], beta=.05)
    l_base = F.smooth_l1_loss(base, target[:, [2, 4]], beta=.05, reduction='none')
    l_base = (l_base * l_base.new_tensor([2., .5])).mean()
    l_support = F.binary_cross_entropy_with_logits(logit, support.to(logit.dtype))
    # A missing entire future contributes to the unknown debit, not to conditional easy harm.
    cond_target = torch.stack(((target[:, 3]-unknown).clamp_min(0.), target[:, 5]), dim=-1)
    e = F.smooth_l1_loss(cond[:, :2], cond_target, beta=.05, reduction='none')
    l_cond = masked_mean((e * e.new_tensor([2., 1.])).mean(dim=-1), support)
    l_unknown = F.smooth_l1_loss(cond[:, 2], unknown, beta=.05)
    return l_gain + l_base + .05*l_support + l_cond + 2.*l_unknown
