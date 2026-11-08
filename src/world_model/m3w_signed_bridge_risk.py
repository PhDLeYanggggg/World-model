"""Predict signed 2% excess for analytic CV-to-OLS4 bridges."""

import numpy as np
from torch import nn

from src.world_model import m3w_reference_bridge_policy as base


def signed_target(moments):
    return moments[..., 2:4]-.02*moments[..., 4:6]


class SignedRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(base.WIDTH, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def replace_bridge_risk(original, signed, scale, correction):
    """Pass signed excess to the existing decoder without changing its threshold."""
    if signed.shape != (len(original), 3, 2) or correction.shape != (3, 2):
        raise ValueError("Require three separately calibrated bridge amplitudes")
    out = original.copy()
    out[:, -3:, 2:4] = (signed+correction[None])*scale[:, None, None]
    out[:, -3:, 4:6] = 0.
    return out


def location_correction(residuals, quantile=.60):
    """Each donor location contributes one residual quantile per arm and risk."""
    available = []
    for r, active in residuals:
        if r.shape[1:] != (3, 2) or r.shape[:2] != active.shape:
            raise ValueError("Donor residuals and action support must align")
        if np.any(active.sum(0) == 0):
            raise ValueError("Donor calibration has no eligible bridge for an amplitude")
        available.append(np.stack([np.quantile(r[active[:, j], j], quantile, axis=0) for j in range(3)]))
    if not available:
        raise ValueError("No donor-only residual evidence")
    return np.stack(available).mean(0)
