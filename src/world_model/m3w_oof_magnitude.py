"""Fitting-locality-only nuisance preprocessing and matched magnitude readout."""
import gzip
import hashlib
import io
import os
from pathlib import Path
import shutil

import numpy as np

from src.world_model.m3w_native_gain_harm import preprocess


def reference_preprocess(x, raw, cv, sites, excluded):
    """Permit exactly one site only for the deepest cross-fitting nuisance head."""
    x, raw, cv, sites = map(np.asarray, (x, raw, cv, sites))
    names = sorted(set(sites))
    if set(names) & set(excluded) or not names:
        raise ValueError('Excluded locality entered reference fitting')
    if len(names) > 1:
        return preprocess(x, raw, cv, sites, next(iter(excluded), '__not_a_site__'))
    if (x.ndim != 2 or raw.shape != (len(x), 2) or cv.shape != (len(x),)
            or sites.shape != (len(x),) or not np.isfinite(x).all()
            or np.isinf(raw).any() or np.isinf(cv).any()
            or not np.array_equal(np.isnan(raw[:, 0]), np.isnan(raw[:, 1]))
            or not np.array_equal(np.isnan(raw[:, 0]), np.isnan(cv))):
        raise ValueError('Aligned supported reference inputs required')
    known = np.isfinite(cv)
    if not known.any() or (raw[known] < 0).any() or (cv[known] < 0).any():
        raise ValueError('Supported nonnegative costs required')
    w = np.zeros(len(x)); w[known] = 1 / known.sum()
    mean = np.zeros(x.shape[1]); second = mean.copy()
    for start in range(0, len(x), 4096):
        z = x[start:start+4096].astype(float); a = w[start:start+4096, None]
        mean += (a*z).sum(0); second += (a*z*z).sum(0)
    scale = float(w[known] @ cv[known]); positive = cv[known & (cv > 0)]
    if scale <= 0 or not len(positive):
        raise ValueError('Positive reference cost scale required')
    return dict(mean=mean, std=np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6),
        known=known, weights=w, cost_scale=scale, constant=(w[known, None]*raw[known]).sum(0),
        training_sites=names, positive_easy_cut=float(np.quantile(positive, .25)),
        hard_cut=float(np.quantile(cv[known], .75)))


def check_lineage(prediction_sites, fitting_sites, outer, producer_sites):
    pred, fit, producer = map(set, (prediction_sites, fitting_sites, producer_sites))
    if not pred or not fit or pred & fit or outer in fit or (pred | fit) & producer:
        raise ValueError('Prediction, fitting and forecast-producer lineage overlaps')


def fit_magnitude(prediction, target, envelope, sites, outer, *, max_slope=8.):
    """Two bounded nonnegative origin slopes; no labels used by predict_magnitude."""
    p, y, env, sites = map(np.asarray, (prediction, target, envelope, sites))
    if (p.shape != y.shape or p.shape != (len(env), 4) or outer in sites
            or sites.shape != env.shape or not np.isfinite(p).all()
            or not np.isfinite(env).all() or (env < 0).any()
            or (p < 0).any() or not np.isfinite(max_slope) or max_slope <= 0):
        raise ValueError('Aligned fitting-only magnitude inputs required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1), ~known) or (y[known] < 0).any():
        raise ValueError('Paired supported targets required')
    valid = known & (env > 0)
    weights = np.zeros(len(p))
    for site in sorted(set(sites)):
        use = valid & (sites == site)
        if not use.any():
            raise ValueError('Each fitting locality needs positive-envelope labels')
        weights[use] = 1/(len(set(sites))*use.sum())
    pp, yy, w = p[valid][:, [1, 3]], y[valid][:, [1, 3]], weights[valid, None]
    denominator = (w*pp**2).sum(0)
    numerator = (w*pp*yy).sum(0)
    slope = np.divide(numerator, denominator, out=np.ones(2), where=denominator > 0)
    return dict(slopes=np.clip(slope, 0, max_slope).tolist(), max_slope=max_slope,
        unbounded_slopes=slope.tolist(), zero_prediction_component=(denominator == 0).tolist(),
        known_rows=int(known.sum()), fitting_rows=int(valid.sum()), training_sites=sorted(set(sites)),
        objective='equal_locality_origin_least_squares_then_nested_envelope_projection',
        fitting_is_calibration_claim=False)


def predict_magnitude(model, prediction, envelope):
    p, env = np.asarray(prediction), np.asarray(envelope)
    if (p.shape != (len(env), 4) or not np.isfinite(p).all()
            or not np.isfinite(env).all() or (env < 0).any() or (p < 0).any()):
        raise ValueError('Finite nonnegative predictions and causal envelope required')
    a = np.asarray(model['slopes'])
    if a.shape != (2,) or not np.isfinite(a).all() or (a < 0).any() or (a > model['max_slope']).any():
        raise ValueError('Invalid fixed nonnegative magnitude coefficients')
    out = p.astype(float).copy()
    out[:, 1] = np.minimum(env, p[:, 1]*a[0])
    out[:, 3] = np.minimum(out[:, 1], p[:, 3]*a[1])
    return out


def compress_checkpoint(directory):
    """Losslessly archive only this run's completed checkpoint; partial fits stay raw."""
    directory = Path(directory); source = directory/'checkpoint.pt'; dest = directory/'checkpoint.pt.gz'
    if not source.exists():
        if not dest.exists(): raise FileNotFoundError(source)
        return dest
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    tmp = dest.with_suffix('.tmp')
    with source.open('rb') as src, tmp.open('wb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0, compresslevel=6) as target:
            shutil.copyfileobj(src, target)
    with gzip.open(tmp, 'rb') as handle:
        if hashlib.sha256(handle.read()).hexdigest() != raw_hash:
            raise ValueError('Checkpoint compression changed bytes')
    os.replace(tmp, dest); source.unlink()
    return dest


def restore(directory):
    import torch
    from src.world_model.m3w_membership_auxiliary import AuxiliaryCostHead
    directory = Path(directory)
    path = directory/'checkpoint.pt.gz'
    if path.exists():
        with gzip.open(path, 'rb') as handle:
            state = torch.load(io.BytesIO(handle.read()), map_location='cpu', weights_only=False)
    else:
        state = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    model = AuxiliaryCostHead(len(state['preprocess']['mean']), state['settings']['width'])
    model.load_state_dict(state['model']); model.eval()
    return model, state
