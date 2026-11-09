"""Dense comparator-relative supervision for an exact-budget scene selector."""

import torch
from torch import nn
from torch.nn import functional as F


TEMPERATURE = .02


class RelativeMargins(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2*width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.head = nn.Linear(32, 7)

    def forward(self, options, anchor):
        return self.head(self.body(torch.cat((options, options-anchor), dim=-1)))


def scores(output, dense):
    # Both arms learn the matched event; only the dense arm learns its margins.
    logp = F.logsigmoid(output[..., 0])
    if dense:
        logp = logp + .5*F.logsigmoid(output[..., 1:3]).sum(-1)
        logp = logp + .25*F.logsigmoid(output[..., 3:]/TEMPERATURE).sum(-1)
    return logp


def loss(output, targets, dense):
    event, _, support = targets[..., :3].unbind(-1)
    result = F.binary_cross_entropy_with_logits(output[..., 0], event, pos_weight=output.new_tensor(5.))
    if dense:
        result = result + .25*F.binary_cross_entropy_with_logits(
            output[..., 1:3], targets[..., 1:3])
        error = F.smooth_l1_loss(output[..., 3:], targets[..., 3:], reduction="none", beta=TEMPERATURE)
        # The easy margin is undefined without an edited easy actor.
        result = result + 4.*(error[..., :3].mean()
                              + (error[..., 3]*support).sum()/support.sum().clamp_min(1.))
    return result
