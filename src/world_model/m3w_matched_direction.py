"""Rotation-only edits and a reflection-aware past context in the edit frame."""

import numpy as np
import torch
from torch import nn


def project_context(history, neighbors, neighbor_mask, displacement, scale):
    """All past positions must be expressed relative to the ego's current origin."""
    h, nh, valid = map(np.asarray, (history, neighbors, neighbor_mask))
    n = len(h)
    if (h.shape != (n, 8, 2) or nh.shape != (n, 4, 8, 2)
            or valid.shape != (n, 4, 8) or valid.dtype != bool
            or not np.isfinite(nh[valid]).all()):
        raise ValueError("Require ego-aligned past and four neighbor masks")
    norms = np.linalg.norm(displacement, axis=-1)
    peak = norms.max(1)
    if (peak <= 0.).any() or np.any(scale <= 0.):
        raise ValueError("Edit and causal normalization must have positive magnitude")
    direction = displacement[np.arange(n), norms.argmax(1)] / peak[:, None]
    side = np.column_stack((-direction[:, 1], direction[:, 0]))
    basis = np.stack((direction, side), axis=-1)
    ego_v = np.diff(h, axis=1)
    ego = (ego_v @ basis / scale[:, None, None]).reshape(n, 14)
    ok = valid.all(2)
    nn = np.where(ok[..., None, None], nh, h[:, None])
    rel = (nn[:, :, -1] - h[:, None, -1]) @ basis / scale[:, None, None]
    recent = (nn[:, :, -1] - nn[:, :, -4])/3.
    dv = (recent-ego_v[:, None, -3:].mean(2)) @ basis / scale[:, None, None]
    distance = np.where(ok, np.linalg.norm(rel, axis=-1), np.inf)
    j = distance.argmin(1)
    count = ok.sum(1)
    nearest = np.where(count > 0, distance[np.arange(n), j], 0.)
    denom = np.maximum(count, 1)[:, None]
    peer = np.column_stack((count/4, rel.sum(1)/denom, nearest,
                            dv.sum(1)/denom, dv[np.arange(n), j]))
    return np.arcsinh(np.column_stack((ego, peer))).astype("float32")


def rotate(edit, angles):
    c, s = angles.cos(), angles.sin()
    x, y = edit.unbind(-1)
    return torch.stack((c*x-s*y, s*x+c*y), dim=-1)


class DirectionHead(nn.Module):
    def __init__(self, width, use_neighbors):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.head = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, edit):
        if not self.use_neighbors:
            features = torch.cat((features[:, :-8], torch.zeros_like(features[:, -8:])), dim=1)
        knots = torch.pi * self.head(features).tanh()
        angles = torch.nn.functional.interpolate(knots[:, None], size=edit.shape[1],
                                                mode="linear", align_corners=True)[:, 0]
        return rotate(edit, angles), angles


def circle_oracle(edit, residual, valid):
    """Hindsight closest point on each fixed-radius circle; never an input."""
    z = np.where(valid[..., None], residual, 0.)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    radius = np.linalg.norm(edit, axis=-1, keepdims=True)
    return np.divide(z*radius, norm, out=edit.copy(), where=norm > 0.)
