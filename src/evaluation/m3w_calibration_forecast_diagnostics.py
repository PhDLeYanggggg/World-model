"""Secondary, query-balanced metrics for a fully frozen calibration roster."""
from collections import defaultdict

import numpy as np

from src.world_model.m3w_record_joint_actions import local_edges


SLICES = ('all', 'easy', 'hard')
MATCHED = ('independent', 'scene', 'pair_additive', 'joint')
CV = 'constant_velocity_causal_fd'


def validate_offsets(offsets, n):
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer) or len(offsets) < 2:
        raise ValueError('Expected a nonempty integer query index')
    if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0):
        raise ValueError('Expected a complete query index')
    return offsets


def query_mass(error, use, offsets):
    n = np.add.reduceat(use.astype(np.int64), offsets[:-1])
    s = np.add.reduceat(np.where(use, error, 0.), offsets[:-1])
    return float((s[n > 0]/n[n > 0]).sum()), int((n > 0).sum())


def individual(paths, target, valid, offsets, *, easy_cut, hard_cut):
    offsets = validate_offsets(offsets, len(valid))
    if valid.shape != target.shape[:2] or not np.isfinite(target[valid]).all():
        raise ValueError('Invalid target mask or labelled positions')
    y = np.where(valid[..., None], target, 0.)
    steps = valid.sum(1)
    known, endpoint = steps > 0, valid[:, -1]

    def error(p):
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError('Forecast shape or support changed')
        e = np.linalg.norm(p-y, axis=2)
        return np.where(valid, e, 0.).sum(1)/np.maximum(steps, 1), e[:, -1]

    floor, _ = error(paths[CV])
    masks = dict(all=known, easy=known & (floor > 0.) & (floor <= easy_cut), hard=known & (floor > hard_cut))
    counts = {k: dict(ade_rows=int(v.sum()), fde_rows=int((v & endpoint).sum())) for k,v in masks.items()}
    out = {}
    for arm, path in paths.items():
        a, f = error(path)
        for name, mask in masks.items():
            out[arm, name, 'ade'] = query_mass(a, mask, offsets)
            out[arm, name, 'fde'] = query_mass(f, mask & endpoint, offsets)
    return out, counts


def graph_queries(origins, offsets):
    offsets = validate_offsets(offsets, len(origins))
    return [local_edges(origins[lo:hi, None])+lo for lo,hi in zip(offsets[:-1],offsets[1:],strict=True)]


def relative(paths, target, valid, edges):
    """Known-only and full-grid bounded joint-minus-control pair ADE gains."""
    if set(paths) != set(MATCHED):
        raise ValueError('Use the four frozen equal-count comparators')
    clean = np.where(valid[..., None], target, 0.)
    out = {k: np.zeros((3,2)) for k in MATCHED[:-1]}
    seen, requested = 0, 0
    for i,j in edges:
        if not len(i):
            continue
        present = valid[i] & valid[j]
        n, size = int(present.sum()), present.size
        seen += n; requested += size
        y = clean[i]-clean[j]
        joint = paths['joint'][i]-paths['joint'][j]
        joint_error = np.linalg.norm(joint-y, axis=2)
        for name, m in out.items():
            p = paths[name][i]-paths[name][j]
            error = np.linalg.norm(p-y, axis=2)
            gain = float((error-joint_error)[present].sum())
            debit = float(np.linalg.norm(p-joint, axis=2)[~present].sum())
            if n:
                m[0] += (gain/n, 1)
            m[1] += ((gain-debit)/size, 1)
            m[2] += ((gain+debit)/size, 1)
    return {(k,t): tuple(m[i]) for k,m in out.items() for i,t in enumerate(('observed','grid_lower','grid_upper'))}, (
        seen, requested)


def aggregate(rows, roster, seeds, *, draws=3000, rng_seed=73107):
    """Merge query means, then average seeds and resample whole source groups."""
    expected = {(r,s) for r in roster for s in seeds}
    if len(rows) != len(expected) or {(r,s) for r,s,_ in rows} != expected:
        raise ValueError('Missing or duplicate recording-seed views')
    keys = list(rows[0][2])
    total = defaultdict(lambda: np.zeros((len(keys), 2)))
    for rec,seed,mass in rows:
        if set(mass) != set(keys):
            raise ValueError('Metric roster changed')
        total[roster[rec],seed] += np.asarray([mass[k] for k in keys])
    groups = sorted(set(roster.values()))
    values = np.stack([total[g,s] for g in groups for s in seeds]).reshape(len(groups),len(seeds),len(keys),2)
    ix = np.random.default_rng(rng_seed).integers(len(groups), size=(draws,len(groups)))

    def interval(x):
        missing = int((~np.isfinite(x)).sum())
        if missing:
            return dict(point=None, interval95=None, missing_source_seed_cells=missing)
        v = x.mean(1)
        return dict(point=float(v.mean()), interval95=np.quantile(v[ix].mean(1), [.025,.975]).tolist(),
                    missing_source_seed_cells=0)

    mass, count = values[...,0], values[...,1]
    estimates = np.divide(mass,count,out=np.full_like(mass,np.nan),where=count > 0)
    results = {key: interval(estimates[...,i]) for i,key in enumerate(keys)}
    lookup = {key:i for i,key in enumerate(keys)}
    against_cv = {k: interval(estimates[...,lookup[(CV,*k[1:])]]-estimates[...,i]) for i,k in enumerate(keys)
                  if len(k) == 3 and k[0] != CV}
    return results, against_cv
