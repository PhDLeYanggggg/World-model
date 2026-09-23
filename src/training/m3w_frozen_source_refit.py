"""Fit-only full-source design for a fixed external predictor refit.

This cannot provide out-of-fold cost targets or held-source evaluation. The
earlier complement implementation stays byte-stable for historical replay.
"""
from __future__ import annotations

import numpy as np


def full_source_design(data, expected_sites):
    sites, valid = np.asarray(data["sites"]), np.asarray(data["valid"])
    expected = list(expected_sites)
    if (len(expected) != len(set(expected)) or set(sites) != set(expected)
            or valid.shape != (len(sites), 12) or valid.dtype != bool):
        raise ValueError("Exact registered fitting sites and aligned twelve-step labels required")
    if (data["geometry"].shape != (len(sites), 476)
            or data["target"].shape != (len(sites), 12, 2)
            or data["scale"].shape != (len(sites),)):
        raise ValueError("Full source row alignment changed")
    factors = np.zeros(len(sites), np.float64)
    groups, normalizers, native_cv = [], {}, []
    for site in sorted(expected):
        ids = np.flatnonzero(sites == site)
        mask, target, scale = valid[ids], data["target"][ids], np.asarray(data["scale"][ids], float)
        if not np.isfinite(target[mask]).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Finite fitting targets and positive causal scales required")
        count = mask.sum(1)
        supported = count > 0
        if not supported.any():
            raise ValueError("A fitting scene has no supported labels")
        baseline = data["geometry"][ids, 332:356].reshape(-1, 12, 2).astype(float)
        distance = np.linalg.norm(baseline - np.where(mask[..., None], target, 0).astype(float), axis=-1)
        error = np.divide(np.where(mask, distance, 0).sum(1), count,
                          out=np.zeros(len(ids)), where=supported)
        mean_cv = float((error[supported] * scale[supported]).mean())
        if mean_cv <= 0 or not np.isfinite(mean_cv):
            raise ValueError("Positive fitting-only CV normalizer required")
        correction = len(ids) / int(supported.sum())
        factors[ids] = correction * scale / mean_cv
        groups.append(ids)
        native_cv.extend((error[supported] * scale[supported]).tolist())
        normalizers[site] = dict(rows=len(ids), supported=int(supported.sum()),
            loss_CV_mean=mean_cv, indexed_to_supported_correction=correction)
    return dict(train_ids=np.arange(len(sites), dtype=np.int64),
        held_ids=np.empty(0, dtype=np.int64), groups=groups, factors=factors,
        normalizers=normalizers, hard_cut=float(np.quantile(native_cv, .75)),
        objective="native_coordinate", evaluation_scope="none_all_rows_are_fitting")
