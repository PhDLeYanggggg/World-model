"""Choice-level learning at a frozen intervention count."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model.m3w_reference_bridge_policy import WIDTH


def choices(legal, count):
    """Flatten row/arm indices; -1 pads the unused slot of a singleton."""
    if count not in (1, 2) or count >= len(legal):
        return np.empty((0, 2), dtype=np.int64)
    row, arm = np.nonzero(legal)
    flat = row*legal.shape[1]+arm
    if count == 1:
        return np.column_stack((flat, np.full(len(flat), -1)))
    i, j = np.triu_indices(len(flat), 1)
    different = row[i] != row[j]
    return np.column_stack((flat[i[different]], flat[j[different]]))


def label_parts(cv, ols, bank, y, valid, cut):
    base = paired(ols, cv, y, valid)[1].sum(1)
    parts = []
    for p in bank.swapaxes(0, 1):
        m = moments(cv, p, y, valid, cut)
        parts.append(np.column_stack((paired(cv, p, y, valid)[1].sum(1),
            paired(ols, p, y, valid)[1].sum(1)-base, m["all_excess"], m["easy_excess"], m["ref_easy"])))
    return np.stack(parts, axis=1), base


def gather(flat, actions):
    value = flat[np.maximum(actions, 0)]
    return np.where((actions >= 0)[..., None], value, 0.)


def utility(parts, actions, ols_offset, scale):
    """Local training proxy, not the recording-level acceptance test."""
    x = gather(parts.reshape(-1, 5), actions).sum(1)/scale
    gain = np.minimum(x[:, 0], x[:, 1]+ols_offset/scale)
    risk = np.maximum(x[:, 2:4], 0.)
    return gain-2.*risk[:, 0]-4.*risk[:, 1]+.025*(x[:, 4] > 0.)


class SwapValue(nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint
        self.encode = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.action = nn.Linear(32, 1, bias=False)
        if joint:
            self.pair = nn.Sequential(nn.Linear(65, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x, pair, query_size):
        z = self.encode(x)
        mask = torch.stack((torch.ones_like(pair), pair), dim=-1)[..., None]
        z = z*mask
        score = self.action(z).sum(dim=-2).squeeze(-1)
        if self.joint:
            interaction = torch.cat((z.sum(dim=-2), (z[..., 0, :]-z[..., 1, :]).abs(), query_size[..., None]), dim=-1)
            score = score+self.pair(interaction).squeeze(-1)*pair
        return score


def objective(score, target, rank):
    if not rank:
        return nn.functional.smooth_l1_loss(score, target, beta=.06)
    p = score[..., 0]-score[..., 1]
    t = target[..., 0]-target[..., 1]
    # Fit the magnitude and sign of a swap without converting near ties to hard labels.
    rel = nn.functional.smooth_l1_loss(p, t, beta=.04)
    rank_loss = nn.functional.binary_cross_entropy_with_logits(p/.06, (t/.06).sigmoid(), reduction="none")
    return rel+.03*(rank_loss*(t.abs()/.06).clamp(max=4.)).mean()
