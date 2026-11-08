"""Causal context and direct counterfactual moments for the nested count menu."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model.m3w_ols_replacement import rollout
from src.world_model.m3w_opendd_velocity_candidate import frames

MIX = np.array([.25, .5, 1.])
PEER_WIDTH = 2*2*17


def contexts(h, row, groups):
    """Use the nearest two observed peers and simultaneous relative histories."""
    rot, scale = frames(h)
    out = np.zeros((len(row), PEER_WIDTH), dtype=np.float32)
    for qi, (ix, _) in enumerate(groups):
        for k, r in enumerate(row[qi]):
            if r < 0:
                continue
            ids = np.arange(ix.start, ix.stop)
            ids = ids[ids != r]
            rel = h[ids]-h[r]
            distance = np.linalg.norm(rel[:, -1], axis=-1)
            # A tied last-point distance is resolved from past geometry, not row ID.
            xy = (rel@rot[r])/(12.*scale[r])
            keys = np.vstack((xy.reshape(len(xy), 16).T[::-1], distance))
            at = np.lexsort(keys)[:2]
            for j, i in enumerate(at):
                start = (2*k+j)*17
                out[qi, start:start+16] = xy[i].reshape(-1).clip(-12., 12.)
                out[qi, start+16] = 1.
    return out


def accumulate(x, row, arm):
    picked = np.where((row >= 0)[..., None], x[row, arm], 0.)
    return np.concatenate((np.zeros_like(picked[:, :1]), np.cumsum(picked, axis=1)), axis=1)


def targets(cv, ols, bank, y, valid, cut, row, arm, scale):
    offset = paired(ols, cv, y, valid)[1].mean(1)
    result = []
    for p in bank.swapaxes(0, 1):
        m = moments(cv, p, y, valid, cut)
        result.append(np.column_stack((paired(cv, p, y, valid)[1].mean(1),
            paired(ols, p, y, valid)[1].mean(1)-offset, m["all_excess"], m["easy_excess"], m["ref_easy"], m["easy_signed"])))
    moments_q = accumulate(np.stack(result, axis=1), row, arm)
    events = np.stack((moments_q[..., 0] > 0., moments_q[..., 2] > 0., moments_q[..., 3] > 0.,
                       moments_q[..., 4] > 0.), axis=-1)
    return np.concatenate(((moments_q/scale[:, None, None]).clip(-8., 8.), events), axis=-1).astype(np.float32)


def past_proxy(h, row, arm, scale):
    """Measure the same interpolation on prefixes of the eight observed positions."""
    observed = []
    for k in (4, 5, 6):
        cv, ols, _, _ = rollout(h[:, :k], 8-k)
        target = h[:, k:]-h[:, k-1:k]
        p = cv[:, None]+MIX[None, :, None, None]*(ols-cv)[:, None]
        g = np.linalg.norm(cv-target, axis=-1).mean(1)[:, None]-np.linalg.norm(p-target[:, None], axis=-1).mean(2)
        observed.append(g)
    v = accumulate(np.mean(observed, axis=0)[..., None], row, arm)[..., 0]
    return v/scale[:, None]


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 20))

    def forward(self, x):
        a = self.net(x).reshape(len(x), 2, 10)
        return torch.cat((a.new_zeros((len(x), 1, 10)), a), dim=1)


def objective(pred, y, legal):
    use = legal.clone()
    use[:, 0] = False
    p, t = pred[use], y[use]
    if not len(p):
        raise ValueError("Fit requires nonzero candidate support")
    err = nn.functional.smooth_l1_loss(p[..., :6], t[..., :6], beta=.1, reduction="none")
    events = nn.functional.binary_cross_entropy_with_logits(p[..., 6:], t[..., 6:])
    return (err*p.new_tensor([1., 1., 2., 3., .5, 1.])).mean()+.05*events


def score(pred, risk=True):
    out = np.minimum(pred[..., 0], pred[..., 1])
    if risk:
        out = out-2.*np.maximum(pred[..., 2], 0.)-4.*np.maximum(pred[..., 3], 0.)-np.maximum(pred[..., 5], 0.)
        out = out+.025*torch.sigmoid(torch.from_numpy(pred[..., 9])).numpy()
    out[:, 0] = 0.
    return out
