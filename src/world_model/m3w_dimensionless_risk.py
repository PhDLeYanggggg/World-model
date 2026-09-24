"""Matched six-moment source heads; input-unit removal is not calibration."""
from __future__ import annotations
import numpy as np
from src.world_model.m3w_net_easy_risk import targets as easy_targets

ARMS = ("native", "dimensionless")
TARGETS = ("benefit_fraction", "harm_fraction", "easy_harm_fraction", "easy_benefit_fraction",
           "easy_denominator_fraction", "easy_probability")


def features(native, distance, scale, arm):
    x, d, s = np.asarray(native), np.asarray(distance), np.asarray(scale)
    if (arm not in ARMS or x.ndim != 2 or x.shape[1] != 356 or d.shape != (len(x),)
            or s.shape != d.shape or not np.isfinite(x).all() or not np.isfinite(d).all()
            or not np.isfinite(s).all() or (d < 0).any() or (s <= 0).any()):
        raise ValueError("Aligned finite frozen features and positive past scale required")
    if arm == "native":
        return x.copy()
    return np.column_stack((x[:, :354], np.log1p(d/s))).astype(np.float32)


def targets(costs, cv, distance, known, cutoff):
    q = easy_targets(costs, cv, distance, known, cutoff)
    total = np.divide(np.where(np.asarray(known)[:, None], costs, 0), np.asarray(distance)[:, None],
                      out=np.zeros_like(costs, dtype=float), where=np.asarray(distance)[:, None] > 0).clip(0, 1)
    out = np.column_stack((total, q))
    validate(out)
    return out


def validate(f):
    f = np.asarray(f)
    if (f.ndim != 2 or f.shape[1] != 6 or not np.isfinite(f).all() or (f < 0).any()
            or (f > 1+2e-6).any() or (f[:, 0]+f[:, 1] > 1+2e-6).any()
            or (f[:, 2] > f[:, 1]+2e-6).any() or (f[:, 3] > f[:, 0]+2e-6).any()
            or (f[:, 2]+f[:, 3] > f[:, 5]+2e-6).any() or (f[:, 4] > f[:, 5]+2e-6).any()):
        raise ValueError("Six coherent bounded fraction targets/predictions required")


def preprocess(x, parent):
    x = np.asarray(x)
    w, known = np.asarray(parent["weights"]), np.asarray(parent["known"])
    if (w.shape != (len(x),) or known.shape != w.shape or known.dtype != bool
            or not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any()
            or w[~known].any() or not np.isclose(w.sum(), 1)):
        raise ValueError("Unchanged supported source-only weighting required")
    mean, second = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for start in range(0, len(x), 4096):
        z = x[start:start+4096].astype(float)
        a = w[start:start+4096, None]
        mean += (a*z).sum(0)
        second += (a*z*z).sum(0)
    return dict(parent, mean=mean, std=np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6))


def scores(fractions, distance, cutoff, history, rho=.02):
    f, d, h = np.asarray(fractions), np.asarray(distance), np.asarray(history)
    validate(f)
    if (d.shape != (len(f),) or h.shape != (len(f), 8, 2) or not np.isfinite(d).all()
            or (d < 0).any() or not np.isfinite(h).all() or not np.isfinite(cutoff)
            or cutoff <= 0 or not 0 <= rho <= 1):
        raise ValueError("Finite causal forecasts and fixed source cutoff required")
    gain = (f[:, 0]-f[:, 1])*d
    risk = (f[:, 2]-f[:, 3])*d
    denominator = f[:, 4]*cutoff
    eligible = (gain > 0) & (d > 0) & (denominator > 0) & np.any(h[:, -1] != h[:, -2], axis=1)
    return dict(gain=gain, risk=risk, denominator=denominator, eligible=eligible,
                point=eligible & (risk <= rho*denominator))
