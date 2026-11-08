"""Estimate risk numerators and reference mass separately before subtraction."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model.m3w_direct_count_value import accumulate


def targets(cv, ols, bank, y, valid, cut, row, arm, scale):
    offset = paired(ols, cv, y, valid)[1].mean(1)
    out = []
    for p in bank.swapaxes(0, 1):
        m = moments(cv, p, y, valid, cut)
        out.append(np.column_stack((paired(cv, p, y, valid)[1].mean(1), paired(ols, p, y, valid)[1].mean(1)-offset,
            np.maximum(-m["diff"], 0.), np.maximum(m["easy_signed"], 0.), np.where(m["known"], 0., m["radius"]),
            m["base"], m["ref_easy"], np.maximum(-m["easy_signed"], 0.))))
    return (accumulate(np.stack(out, axis=1), row, arm)/scale[:, None, None]).clip(-8., 8.).astype(np.float32)


class Factorized(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 28))
        with torch.no_grad():
            self.net[-1].bias.view(2, 14)[:, 8:] = -4.

    def forward(self, x):
        v = self.net(x).reshape(len(x), 2, 14)
        return torch.cat((v.new_zeros((len(x), 1, 14)), v), dim=1)


def objective(pred, truth, legal):
    use = legal.clone()
    use[:, 0] = False
    p, t = pred[use], truth[use]
    gain = nn.functional.smooth_l1_loss(p[:, :2], t[:, :2], beta=.1)
    target = t[:, 2:]
    pos = target > 0.
    prob = nn.functional.binary_cross_entropy_with_logits(p[:, 2:8], pos.float())
    size = nn.functional.softplus(p[:, 8:])
    cond = nn.functional.smooth_l1_loss(size, target, beta=.1, reduction="none")
    cond = ((cond*pos).sum(0)/pos.sum(0).clamp_min(1)).mean()
    marginal = nn.functional.smooth_l1_loss(size*p[:, 2:8].sigmoid(), target, beta=.1)
    return gain+cond+.5*marginal+.05*prob


def moments_for_score(pred):
    v = (pred[..., 2:8].sigmoid()*nn.functional.softplus(pred[..., 8:])).numpy()
    harm, eh, unknown, base, easy, benefit = np.moveaxis(v, -1, 0)
    eh, easy = np.minimum(eh, harm), np.minimum(easy, base)
    benefit = np.minimum(benefit, easy)
    out = np.zeros((*v.shape[:2], 10), dtype=np.float32)
    out[..., :2] = pred[..., :2].numpy()
    out[..., 2] = harm+unknown-.02*base
    out[..., 3] = eh+unknown-.02*easy
    out[..., 4] = easy
    out[..., 5] = eh-benefit
    out[..., 9] = pred[..., 6].numpy()
    out[:, 0] = 0.
    return out
