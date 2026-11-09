"""Fit causal action costs and compare additive with joint exact-slot decisions."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model import m3w_slate_shared_menu as shared

AMPLITUDES = (.25, .5, 1.)


def training_examples(h, cv, ols, bank, y, valid, easy_cut):
    candidate = cv[:, None] + np.concatenate([a*(bank-cv[:, None]) for a in AMPLITUDES], axis=1)
    x, scale, legal, _ = shared.costs.features(h, cv, ols, candidate)
    _, _, gain, _ = candidate_terms(cv, ols, candidate, y, valid)
    risk = risk_excess(cv, candidate, y, valid, easy_cut)
    cv_ade, _ = errors(cv, y, valid)
    easy = valid.any(1) & (cv_ade > 0.) & (cv_ade <= easy_cut)
    signed = np.column_stack([np.where(easy, errors(p, y, valid)[0]-cv_ade, 0.) for p in candidate.swapaxes(0, 1)])
    values = np.concatenate((gain/valid.shape[1], risk, signed[..., None]), axis=-1)/scale[:, None, None]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite candidate cost target')
    return torch.from_numpy(x[legal]), torch.from_numpy(values[legal].astype(np.float32))


class CostHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(shared.costs.WIDTH, 48), nn.SiLU(), nn.Linear(48, 24),
                                 nn.SiLU(), nn.Linear(24, 5))

    def forward(self, x):
        return self.net(x)


def utility(value):
    return value[..., :2].min(axis=-1) - np.maximum(value[..., 2:], 0.).sum(axis=-1)


def ranks(value, options):
    """Same cost forecasts; combine risk before or after evaluating the options."""
    live = options >= 0
    costs = np.where(live[..., None], value[np.maximum(options, 0)], 0.)
    return utility(costs).sum(axis=1), utility(costs.sum(axis=1))


@torch.inference_mode()
def choose(net, h, cv, ols, bank, radii):
    paths, slots = shared.slot_menu(cv, bank, radii)
    options = shared.legal_assignments(slots[0][0], slots[1][0] if len(slots) == 2 else None)
    row = np.concatenate([r for r, _ in slots])
    arm = np.concatenate([c+i*bank.shape[1] for i, (_, c) in enumerate(slots)])
    x, scale, legal, _ = shared.costs.features(h, cv, ols, paths)
    if not len(options) or not legal[row, arm].all():
        raise ValueError('Frozen peak slots lack legal choices')
    value = net(torch.from_numpy(x[row, arm])).numpy().astype(float) * scale[row, None]
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite predicted action costs')
    result = []
    for scores in ranks(value, options):
        take = options[int(scores.argmax())]
        take = take[take >= 0]
        p = cv.copy()
        p[row[take]] = paths[row[take], arm[take]]
        peak = np.linalg.norm(p-cv, axis=-1).max(1)
        np.testing.assert_allclose(np.sort(peak[peak > 0.]), np.sort(radii), atol=1e-9, rtol=1e-9)
        result.append(p)
    if len(radii) == 1:
        np.testing.assert_array_equal(*result)
    return result
