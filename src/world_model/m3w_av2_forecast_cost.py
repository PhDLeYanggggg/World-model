"""Learn decision costs on a fixed full grid; future masks only form labels."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_slate_relative_gain as relative

MODES = ("history", "known", "debit", "debit_easy")
COMPONENTS = ("trajectory_known", "trajectory_debit", "pair_known", "pair_debit", "signed_easy_excess")


def describe(history, cv, ols, paths, anchor, cutoff):
    x, _, pair_bound, unit, _ = relative.features(history, cv, ols, paths, anchor)
    distance = np.linalg.norm(paths-anchor, axis=-1)/unit
    edit = np.linalg.norm(paths-cv, axis=-1)/unit
    trajectory_bound = distance.mean((1, 2))
    easy_bound = edit.max(2).mean(1)+.02*cutoff/unit
    bounds = np.column_stack((trajectory_bound, trajectory_bound, pair_bound, pair_bound, easy_bound))
    blocks = distance.reshape(len(paths), len(cv), 3, 4).mean((1, 3))
    features = np.column_stack((x, np.log1p(blocks), np.log1p(bounds[:, [0, 2, 4]])))
    return features.astype(np.float32), bounds.astype(np.float32), unit


def targets(paths, anchor, cv, target, valid, cutoff, unit):
    """Keep observed gain and missing debit on the same, outcome-free denominator."""
    safe = np.where(valid[..., None], target, 0.)
    norm = lambda value: np.linalg.norm(value, axis=-1)
    err, ref, baseline = norm(paths-safe), norm(anchor-safe), norm(cv-safe)
    known = np.where(valid, ref-err, 0.).mean((1, 2))/unit
    debit = np.where(valid, 0., norm(paths-anchor)).mean((1, 2))/unit
    pair = relative.labels(paths, anchor, safe, valid, unit)
    n = valid.sum(1).clip(1)
    cv_ade = np.where(valid, baseline, 0.).sum(1)/n
    easy = (cv_ade > 0.) & (cv_ade <= cutoff)
    delta = np.where(valid, err-baseline, 0.).sum(2)/n
    signed = np.where(easy, delta-.02*cv_ade, 0.).mean(1)/unit
    return np.column_stack((known, debit, pair, signed)).astype(np.float32)


class CostHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32),
                                    nn.SiLU(), nn.Linear(32, len(COMPONENTS)))

    def forward(self, x):
        z = self.layers(x)
        return torch.stack((z[:, 0].tanh(), z[:, 1].sigmoid(), z[:, 2].tanh(),
                            z[:, 3].sigmoid(), z[:, 4].tanh()), dim=1)


def scores(prior, ratios, bounds):
    """Mask cost terms in the same fitted network; weights are not tuned on outcomes."""
    tk, td, pk, pd, easy = ratios.unbind(1)
    # A zero geometric difference is a genuine zero, not a signed prediction.
    tk, td = (t*(bounds[:, 0] > 0.) for t in (tk, td))
    pk, pd = (t*(bounds[:, 2] > 0.) for t in (pk, pd))
    known = prior*(4.*tk).sigmoid()*(4.*pk).sigmoid()
    debit = prior*(4.*(tk-td)).sigmoid()*(4.*(pk-pd)).sigmoid()
    return dict(zip(MODES, (prior, known, debit, debit*(-4.*easy).sigmoid()), strict=True))
