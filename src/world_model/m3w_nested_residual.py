"""Locality-excluded residual producers with fixed matched in-sample controls."""
import numpy as np
from src.world_model.m3w_native_gain_harm import preprocess
from src.evaluation.m3w_harm_tail_diagnostics import event_targets

VARIANTS = ('oof', 'in_sample_next', 'in_sample_prev')


def fitting_sites(sites, outer):
    sites = np.asarray(sites)
    names = sorted(set(sites))
    if sites.ndim != 1 or len(names) != 3 or outer in names:
        raise ValueError('Exactly three fitting localities, outer excluded')
    return names


def labels(raw, cv, cut):
    raw, cv = np.asarray(raw, float), np.asarray(cv, float)
    if raw.shape != (len(cv), 2):
        raise ValueError('Reference cost and all-harm labels required')
    return event_targets(np.column_stack((raw, np.zeros_like(raw))), cv, cut)


def inner_inputs(x, raw, cv, sites, outer, inner):
    names = fitting_sites(sites, outer)
    if inner not in names:
        raise ValueError('Inner-held locality must belong to fitting roster')
    x, raw, cv, sites = map(np.asarray, (x, raw, cv, sites))
    take = sites != inner
    pr = preprocess(x[take], raw[take], cv[take], sites[take], inner)
    y = labels(raw[take], cv[take], pr['positive_easy_cut'])
    easy = np.where(pr['known'], (cv[take] > 0) & (cv[take] <= pr['positive_easy_cut']), np.nan)
    assert outer not in pr['training_sites'] and inner not in pr['training_sites']
    return take, pr, y, easy


def assemble(bank, raw, cv, sites, outer, variant):
    """One equally sized producer per row; only residual provenance differs."""
    names = fitting_sites(sites, outer)
    if set(bank) != set(names) or variant not in VARIANTS:
        raise ValueError('Complete registered producer bank and variant required')
    sites = np.asarray(sites); n = len(sites)
    p, y = np.empty((n, 4)), np.empty((n, 4))
    cuts = np.empty(n); producers = np.empty(n, dtype=sites.dtype)
    offset = {'oof':0, 'in_sample_next':1, 'in_sample_prev':-1}[variant]
    for j, site in enumerate(names):
        producer = names[(j+offset) % 3]; model = bank[producer]
        if sorted(model['training_sites']) != [s for s in names if s != producer]:
            raise ValueError('Inner fitting lineage mismatch')
        if (site in model['training_sites']) != (variant != 'oof'):
            raise ValueError('Residual producer exposure mismatch')
        pred = np.asarray(model['prediction'])
        if (pred.shape != (n, 4) or not np.isfinite(pred).all() or (pred < 0).any()
                or (pred[:,3] > pred[:,1]).any()):
            raise ValueError('Aligned finite producer predictions required')
        use = sites == site; cut = model['cut']
        p[use] = pred[use]; y[use] = labels(np.asarray(raw)[use], np.asarray(cv)[use], cut)
        cuts[use] = cut; producers[use] = producer
    return p, y, cuts, producers


def cut_drift(cv, cuts, outer_cut):
    cv, cuts = np.asarray(cv), np.asarray(cuts)
    known = np.isfinite(cv)
    if cuts.shape != cv.shape or not np.isfinite(cuts).all() or (cuts <= 0).any() or outer_cut <= 0:
        raise ValueError('Positive aligned fitting-only cuts required')
    a = (cv[known] > 0) & (cv[known] <= cuts[known])
    b = (cv[known] > 0) & (cv[known] <= outer_cut)
    return dict(known_rows=int(known.sum()), easy_label_disagreement_fraction=float(np.mean(a != b)),
                inner_to_outer_cut_ratio_range=[float(cuts.min()/outer_cut), float(cuts.max()/outer_cut)])
