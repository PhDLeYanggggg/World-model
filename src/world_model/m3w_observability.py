"""Past box geometry and per-step observability for selective forecasting."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("direct_motion", "direct_quality", "factor_motion", "factor_quality")
NAMES = ("width", "height", "aspect", "width_jitter", "height_jitter",
         "center_jitter", "border_clearance", "projected_border", "border_trend",
         "current_overlap", "current_neighbors", "neighbor_past_support")


def box_features(box, valid, target, size):
    """Exactly eight past boxes; target order is the existing roster order."""
    if box.ndim != 3 or box.shape[1:] != (8, 4) or valid.shape != box.shape[:2]:
        raise ValueError("Require eight observed-frame box slots")
    if not valid[target].all():
        raise ValueError("Quality features must follow the complete-history roster")
    width, height = np.asarray(size, dtype=float)
    if min(width, height) <= 0 or (box[valid][:, 2:] <= 0).any():
        raise ValueError("Missing box or source dimensions")
    b = box[target]
    wh = b[..., 2:]
    center = b[..., :2] + wh / 2
    logwh = np.log(wh)
    v = np.diff(center, axis=1)
    jitter = np.linalg.norm(v - np.median(v, axis=1, keepdims=True), axis=-1).mean(1)
    scale = np.sqrt(wh[..., 0] * wh[..., 1]).mean(1).clip(1.)
    sides = np.stack((b[..., 0], width-b[..., 0]-wh[..., 0],
                      b[..., 1], height-b[..., 1]-wh[..., 1]), axis=-1)
    clearance = (sides / wh[..., [0, 0, 1, 1]]).min(-1)
    time = np.arange(4.) - 1.5
    slope = np.einsum("ntd,t->nd", center[:, -4:] - center[:, -1:], time) / (time @ time)
    projected = center[:, -1] + 12. * slope
    p = np.column_stack((projected[:, 0], width-projected[:, 0], projected[:, 1], height-projected[:, 1]))
    p = (p / wh[:, -1, [0, 0, 1, 1]] - .5).min(1)
    current = np.flatnonzero(valid[:, -1])
    now, peers = b[:, -1], box[current, -1]
    intersection = np.maximum(0., np.minimum(now[:, None, :2]+now[:, None, 2:],
        peers[None, :, :2]+peers[None, :, 2:]) - np.maximum(now[:, None, :2], peers[None, :, :2])).prod(-1)
    is_peer = target[:, None] != current
    iou = intersection / (now[:, None, 2:].prod(-1)+peers[None, :, 2:].prod(-1)-intersection).clip(1e-9)
    peer_center = peers[:, :2] + peers[:, 2:] / 2
    neighbors = is_peer & (np.linalg.norm(center[:, -1, None] - peer_center, axis=-1) <= 4. * scale[:, None])
    count = neighbors.sum(1)
    support = (neighbors @ valid[current].mean(1)) / count.clip(1)
    return np.column_stack((np.log1p(1000.*wh[:, -1, 0]/width), np.log1p(1000.*wh[:, -1, 1]/height),
        logwh[:, -1, 0]-logwh[:, -1, 1], np.abs(np.diff(logwh, axis=1)).mean(1),
        jitter/scale, clearance[:, -1], p, clearance[:, -1]-clearance[:, 0],
        (iou * is_peer).max(1), np.log1p(count), support)).clip(-16., 16.).astype(np.float32)


def components(floor, delta, target, observed, scale):
    clean = np.where(observed[..., None], target, 0.)
    d = np.linalg.norm(delta, axis=-1)
    gain = np.linalg.norm(floor-clean, axis=-1) - np.linalg.norm(floor+delta-clean, axis=-1)
    d_norm = d / scale[:, None]
    fractional = np.divide(gain, d, out=np.zeros_like(d), where=d > 1e-8).clip(-1., 1.)
    signed = np.where(observed, fractional, -1.) * d_norm
    return d_norm.astype(np.float32), fractional.astype(np.float32), signed.mean(1).astype(np.float32)


def network(width):
    net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 25))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def estimate(output, debit, arm):
    p, mu = output[..., 1:13].sigmoid(), output[..., 13:].tanh()
    direct = output[..., 0]
    u = ((p*mu - (1-p))*debit).mean(-1)
    return (u if arm.startswith("factor") else direct), p


def loss(output, debit, fractional, observed, direct_target, arm):
    if arm not in ARMS:
        raise ValueError("Unknown ablation arm")
    support = F.binary_cross_entropy_with_logits(output[..., 1:13], observed.float())
    if arm.startswith("direct"):
        return F.smooth_l1_loss(output[..., 0], direct_target, beta=.02) + .05 * support
    known = F.smooth_l1_loss(output[..., 13:].tanh(), fractional, beta=.2, reduction="none")
    error = (known*observed*debit).mean()
    return error + .05*support
