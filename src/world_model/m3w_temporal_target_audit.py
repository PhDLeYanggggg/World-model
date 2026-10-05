"""TRAIN-only temporal probes and masked auxiliary supervision, not a policy."""
import hashlib

import numpy as np

from src.world_model import m3w_source_forest as forest


def targets(reference, candidate, target, valid):
    r, p, y = [np.asarray(a, float) for a in (reference, candidate, target)]
    m = np.asarray(valid)
    if (r.ndim != 3 or r.shape[1:] != (12, 2) or p.shape != r.shape or y.shape != r.shape
            or m.shape != r.shape[:2] or m.dtype != bool or not np.isfinite(r).all()
            or not np.isfinite(p).all() or not np.isfinite(y[m]).all()):
        raise ValueError('Twelve aligned steps and explicit observed-target mask required')
    safe = np.where(m[..., None], y, 0.)
    re = np.linalg.norm(r-safe, axis=2)
    delta = np.linalg.norm(p-safe, axis=2)-re
    n = m.sum(1)

    def average(a):
        return np.divide(np.where(m, a, 0.).sum(1), n,
                         out=np.full(len(m), np.nan), where=n > 0)

    signed = average(delta)
    b, h = average(np.maximum(-delta, 0)), average(np.maximum(delta, 0))
    cancel = np.minimum(b, h)
    series = np.where(m[..., None], np.stack((delta, re), 2), np.nan)
    out = dict(valid_steps=n, signed=signed, reference=average(re),
               benefit=np.maximum(-signed, 0), harm=np.maximum(signed, 0),
               gross_benefit=b, gross_harm=h, cancellation=cancel,
               opposite_sign=(np.where(m, delta, 0).max(1) > 0)
                             & (np.where(m, delta, 0).min(1) < 0))
    return series, out


def validate_series(series):
    v = np.asarray(series, float)
    if v.ndim != 3 or v.shape[1:] != (12, 2) or np.isinf(v).any():
        raise ValueError('Aligned finite or missing temporal labels required')
    mask = np.isfinite(v).all(2)
    if not (mask | np.isnan(v).all(2)).all() or (v[..., 1][mask] < 0).any():
        raise ValueError('Both channels share the explicit missing support')
    return v, mask


def fit(state, x, envelope, series, sites, recordings, frames):
    """Analytic TRAIN means on frozen leaves; no validation labels accepted."""
    v, mask = validate_series(series)
    z, _ = forest.causal_inputs(x, envelope, state['preprocess'])
    if len(v) != len(z): raise ValueError('Aligned TRAIN rows required')
    known = mask.any(1)
    w, _ = forest.core.weights(sites, recordings, frames, known)
    if not known.any(): raise ValueError('No known TRAIN labels')
    v = v / state['preprocess']['scale']
    safe = np.where(mask[..., None], v, 0)
    den = (w[:, None]*mask).sum(0)
    total = (w[:, None, None]*safe).sum(0)
    glob = np.divide(total, den[:, None], out=np.full((12, 2), np.nan), where=den[:, None] > 0)
    rowmean = np.divide(safe.sum(1), mask.sum(1)[:, None],
                        out=np.zeros((len(z), 2)), where=mask.sum(1)[:, None] > 0)
    tables = []
    for tree in state['model'].estimators_:
        leaves = tree.apply(z); nodes, inv = np.unique(leaves, return_inverse=True)
        k = len(nodes)
        d = np.stack([np.bincount(inv, weights=w*mask[:, t], minlength=k) for t in range(12)], 1)
        num = np.stack([np.stack([np.bincount(inv, weights=w*safe[:, t, c], minlength=k)
                                 for c in range(2)], 1) for t in range(12)], 1)
        means = np.divide(num, d[..., None], out=np.broadcast_to(glob, num.shape).copy(),
                          where=d[..., None] > 0)
        rd = np.bincount(inv, weights=w, minlength=k)
        rn = np.stack([np.bincount(inv, weights=w*rowmean[:, c], minlength=k) for c in range(2)], 1)
        rm = np.divide(rn, rd[:, None], out=np.broadcast_to(w@rowmean, rn.shape).copy(), where=rd[:, None] > 0)
        tables.append(dict(nodes=nodes, temporal=means, constant=rm, supported=d > 0))
    return dict(global_mean=glob, tables=tables, scale=float(state['preprocess']['scale']))


def predict(state, fitted, x, envelope):
    """No future masks, labels or record length at inference."""
    z, _ = forest.causal_inputs(x, envelope, state['preprocess'])
    trees = state['model'].estimators_
    if len(trees) != len(fitted['tables']): raise ValueError('Frozen routing mismatch')
    n = len(z); temporal = np.zeros((n, 12, 2)); constant = np.zeros_like(temporal)
    support = np.zeros((n, 12))
    for tree, table in zip(trees, fitted['tables']):
        leaves = tree.apply(z); ix = np.searchsorted(table['nodes'], leaves)
        if (ix >= len(table['nodes'])).any() or not np.array_equal(table['nodes'][ix], leaves):
            raise ValueError('Query routed to a leaf absent from the fixed TRAIN population')
        temporal += table['temporal'][ix]/len(trees)
        constant += table['constant'][ix, None]/len(trees)
        support += table['supported'][ix]/len(trees)
    global_mean = np.broadcast_to(fitted['global_mean'], temporal.shape).copy()
    # Unobserved TRAIN steps remain unsupported, even for the constant comparator.
    absent = ~np.isfinite(fitted['global_mean']).all(1)
    constant[:, absent] = np.nan
    return dict(temporal_leaf=temporal, rowmean_leaf=constant, global_temporal=global_mean), support


def fingerprint(fitted):
    h = hashlib.sha256()
    h.update(forest.fingerprint(fitted['global_mean']).encode())
    h.update(str(fitted['scale']).encode())
    for table in fitted['tables']:
        for key in ('nodes', 'temporal', 'constant', 'supported'):
            h.update(forest.fingerprint(table[key]).encode())
    return h.hexdigest()


def score(predictions, series, sites, recordings, frames):
    truth, valid = validate_series(series)
    common = valid.copy()
    if set(predictions) != {'temporal_leaf', 'rowmean_leaf', 'global_temporal'}:
        raise ValueError('All three frozen probe controls required')
    for p in predictions.values():
        if p.shape != truth.shape or np.isinf(p).any(): raise ValueError('Aligned probe predictions required')
        common &= np.isfinite(p).all(2)
    known = common.any(1)
    result = dict(rows=len(truth), unknown_rows=int((~valid.any(1)).sum()),
                  observed_steps=int(valid.sum()), scored_steps=int(common.sum()),
                  unsupported_observed_steps=int((valid & ~common).sum()))
    if not known.any():
        return dict(result, **{k: None for k in predictions})
    w, _ = forest.core.weights(sites, recordings, frames, known)
    for name, p in predictions.items():
        err = np.where(common[..., None], p-truth, 0.)**2
        row = np.divide(err.sum(1), common.sum(1)[:, None], out=np.zeros((len(truth), 2)),
                        where=common.sum(1)[:, None] > 0)
        result[name] = (w@row).tolist()
    return result


def auxiliary_loss(prediction, target, valid, segments, query_count):
    """Loss only: keep the primary five-moment objective and deployment API intact."""
    import torch
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (12, 2)
            or valid.shape != prediction.shape[:2] or valid.dtype != torch.bool
            or segments.shape != (len(prediction),) or segments.dtype != torch.int64
            or query_count <= 0 or len(prediction) == 0 or segments.min() < 0
            or segments.max() >= query_count or not torch.isfinite(prediction).all()
            or not torch.isfinite(target[valid]).all()):
        raise ValueError('Aligned temporal supervision and nonempty query IDs required')
    counts = valid.sum(1); known = counts > 0
    # Replace missing labels before arithmetic so NaNs cannot poison masked gradients.
    truth = torch.where(valid[..., None], target.detach(), torch.zeros_like(target))
    delta = torch.where(valid[..., None], prediction-truth, torch.zeros_like(prediction))
    rows = delta.square().sum((1, 2))/(2*counts.clamp_min(1))
    sizes = torch.bincount(segments[known], minlength=query_count)
    if (sizes == 0).any(): raise ValueError('Each sampled query needs observed supervision')
    total = rows.new_zeros(query_count).index_add(0, segments[known], rows[known])
    return (total/sizes).mean()
