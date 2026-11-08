"""Past-only bridge costs and a two-intervention query-local optimizer."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.world_model import m3w_ols_replacement as motion
from src.world_model.m3w_opendd_velocity_candidate import frames

WIDTH = 45


def features(h, cv, ols, bank):
    rot, scale = frames(h)
    world = scale*12.
    past = ((h[:, :7]-h[:, -1:]) @ rot)/scale[:, None, None]
    refs = np.concatenate(((cv[:, -1:] @ rot)[:, 0], (ols[:, -1:] @ rot)[:, 0]), axis=1)/world[:, None]
    probe = []
    for k in (4, 5, 6):
        a, b, _, _ = motion.rollout(h[:, :k], 8-k)
        y = h[:, k:]-h[:, k-1:k]
        ea, eb = [np.linalg.norm(p-y, axis=-1).mean(1) for p in (a, b)]
        probe.append((ea-eb)/scale)
    g = np.stack(probe, axis=1)
    common = np.column_stack((past.reshape(len(h), 14), refs, g.mean(1), g.min(1)))
    off = bank-cv[:, None]
    r = np.linalg.norm(off, axis=-1).max(2)
    u = (off @ rot[:, None])/np.where(r > 0., r, 1.)[..., None, None]
    x = np.concatenate((np.broadcast_to(common[:, None], (*r.shape, common.shape[1])),
                        np.log1p(r/world[:, None])[..., None], u.reshape(*r.shape, 24)), axis=-1)
    if x.shape[-1] != WIDTH or not np.isfinite(x).all():
        raise ValueError("Invalid causal bridge features")
    return x.astype(np.float32), world, r > 0., g.mean(1)


def targets(cv, ols, bank, y, valid, cut, scale):
    base = paired(ols, cv, y, valid)[1].mean(1)
    values = []
    for a in bank.swapaxes(0, 1):
        m = moments(cv, a, y, valid, cut)
        gain = paired(cv, a, y, valid)[1].mean(1)
        other = paired(ols, a, y, valid)[1].mean(1)-base
        harm = m["all_excess"]+BUDGET*m["base"]
        easy_harm = m["easy_excess"]+BUDGET*m["ref_easy"]
        values.append(np.column_stack((gain, other, harm.clip(0), easy_harm.clip(0), m["base"], m["ref_easy"]))/scale[:, None])
    return np.stack(values, axis=1).astype(np.float32)


class BridgeValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WIDTH, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 6))
        with torch.no_grad():
            self.net[-1].bias[2:] = -4.

    def forward(self, x):
        a = self.net(x)
        return torch.cat((a[..., :2], nn.functional.softplus(a[..., 2:])), dim=-1)


def loss(out, target, legal):
    v = nn.functional.smooth_l1_loss(out[legal], target[legal], beta=.05, reduction="none")
    return (v*out.new_tensor([1., 1., 2., 2., .5, 1.])).mean()


def query_choices(cost, legal, queries, motion_score):
    """Maximize two-reference increment under a predicted local risk budget."""
    if not np.isfinite(cost).all():
        raise ValueError("Expected finite predicted costs")
    result = {k: np.full(len(cost), -1, dtype=int) for k in ("joint", "independent", "matched_independent", "matched_motion", "bridge_only")}
    utility = cost[..., :2].min(2)
    risk = cost[..., 2:4]-BUDGET*cost[..., 4:6]
    for q in np.unique(queries):
        ix = np.flatnonzero(queries == q)
        cap = min(2, len(ix)-1)
        if cap <= 0:
            continue
        a = np.where(legal[ix], utility[ix], -np.inf)
        best = a.argmax(1)
        value = a[np.arange(len(ix)), best]
        order = np.argsort(-value, kind="stable")
        selected = order[:cap][value[order[:cap]] > 0.]
        result["independent"][ix[selected]] = best[selected]
        g, k = np.nonzero(legal[ix])
        r, u = risk[ix[g], k], cost[ix[g], k, :2]
        for name in ("joint", "bridge_only"):
            allow = np.ones(len(g), dtype=bool) if name == "joint" else (k >= cost.shape[1]-3)
            score = np.where(allow & (r <= 0.).all(1), u.min(1), -np.inf)
            action = [int(score.argmax())] if len(score) and score.max() > 0. else []
            hi = score[action[0]] if action else 0.
            if cap == 2 and len(g) > 1:
                i, j = np.triu_indices(len(g), 1)
                ok = (g[i] != g[j]) & allow[i] & allow[j] & ((r[i]+r[j]) <= 0.).all(1)
                benefit = np.where(ok, (u[i]+u[j]).min(1), -np.inf)
                at = int(benefit.argmax())
                if benefit[at] > hi:
                    action = [i[at], j[at]]
            if action:
                result[name][ix[g[action]]] = k[action]
            if name != "joint" or not action:
                continue
            match = order[:len(action)]
            result["matched_independent"][ix[match]] = best[match]
            # Keep the selected family counts, not merely the number of interventions.
            available = np.ones(len(ix), dtype=bool)
            for arm in sorted(k[action], key=lambda x: int(legal[ix, x].sum())):
                candidates = np.flatnonzero(available & legal[ix, arm])
                if not len(candidates):
                    break
                t = candidates[np.argmax(motion_score[ix[candidates]])]
                result["matched_motion"][ix[t]] = arm
                available[t] = False
            if (result["matched_motion"][ix] >= 0).sum() != len(action):
                result["matched_motion"][ix] = result["joint"][ix]
    return result
