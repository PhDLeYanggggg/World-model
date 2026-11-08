"""An 8/12 fixed-head EqMotion control in a focal comoving frame."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_eqmotion_adapter import EqMotionFixedHead


ARMS = ("centered_peers", "comoving_peers", "comoving_ego")
ARCH = dict(history_steps=8, prediction_steps=12, hidden_nf=32, channels=32,
            layers=2, fixed_head=0, prune_unused_heads=True)
NEIGHBORS = 3


def context(history, offsets):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Require finite native eight-step histories")
    if offsets[0] != 0 or offsets[-1] != len(h) or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid query roster")
    ego = h - h[:, -1:]
    peers = np.zeros((len(h), NEIGHBORS, 8, 2))
    mask = np.zeros(peers.shape[:2], dtype=bool)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        x = h[lo:hi]
        distance = np.square(x[:, None, -1] - x[None, :, -1]).sum(-1)
        np.fill_diagonal(distance, np.inf)
        k = min(NEIGHBORS, hi-lo-1)
        j = np.argsort(distance, axis=1, kind="stable")[:, :k]
        peers[lo:hi, :k] = x[j] - x[:, None, -1:]
        mask[lo:hi, :k] = True
    speed_scale = np.sqrt(np.square(np.diff(h, axis=1)).sum(2).mean(1)).clip(.05)
    return tuple(torch.from_numpy(a) for a in (ego.astype(np.float32), peers.astype(np.float32),
                                               mask, speed_scale.astype(np.float32)))


def frame(ego, peers, mask, arm):
    if arm not in ARMS:
        raise ValueError("Unknown comoving control")
    past = torch.arange(-7, 1, dtype=ego.dtype, device=ego.device)
    future = torch.arange(1, 13, dtype=ego.dtype, device=ego.device)
    v = ego[:, -1] - ego[:, -2]
    shift = v[:, None, :]*past[None, :, None]
    cv = v[:, None, :]*future[None, :, None]
    h = ego-shift
    p = torch.where(mask[..., None, None], peers-shift[:, None], 0.)
    # A single observation-only scale for all three arms; invariant to a common velocity boost.
    scale = h.norm(dim=-1).amax(1).clamp_min(1.)
    if arm == "centered_peers":
        h, p = ego, peers
        restore, baseline = torch.zeros_like(cv), cv
    else:
        restore, baseline = cv, torch.zeros_like(cv)
    if arm == "comoving_ego":
        mask = torch.zeros_like(mask)
        p = torch.zeros_like(p)
    inputs = dict(history=torch.cat((h/scale[:, None, None], (past/12).expand(len(h), -1)[..., None]), -1),
        neighbors=torch.cat((p/scale[:, None, None, None], (past/12).expand(*p.shape[:3])[..., None]), -1),
        history_mask=torch.ones(h.shape[:2], dtype=torch.bool, device=h.device),
        neighbor_mask=mask[..., None].expand(p.shape[:3]),
        baseline=baseline/scale[:, None, None], prediction_time=(future/12).expand(len(h), -1),
        request_mask=torch.ones(cv.shape[:2], dtype=torch.bool, device=h.device))
    return inputs, scale, restore


class Forecast(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.predictor = EqMotionFixedHead(**ARCH)

    def forward(self, ego, peers, mask):
        inputs, scale, restore = frame(ego, peers, mask, self.arm)
        return self.predictor(inputs)*scale[:, None, None] + restore


def loss(net, batch, y, valid):
    ego, peers, mask, scale = batch
    p = net(ego, peers, mask)
    residual = torch.where(valid[..., None], p-y, 0.)/scale[:, None, None]
    error = (residual.square().sum(-1)+.0025).sqrt()-.05
    return (error.sum(1)/valid.sum(1).clamp_min(1)).mean()
