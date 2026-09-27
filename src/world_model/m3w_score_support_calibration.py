"""Frozen-score source calibration, not a conformal risk certificate."""
from itertools import combinations
import numpy as np


def role_pairs(producer, controller, outer):
    groups = [set(g) for g in (producer, controller, outer)]
    if any(len(g) != 4 for g in groups) or len(set.union(*groups)) != 12:
        raise ValueError('Three disjoint four-source roles required')
    return [(list(c), sorted(groups[2]-set(c))) for c in combinations(sorted(outer), 2)]


def support_distance(x, mean, std):
    x, mean, std = map(lambda a: np.asarray(a, float), (x, mean, std))
    if (x.ndim != 2 or mean.shape != (x.shape[1],) or std.shape != mean.shape
            or any(not np.isfinite(a).all() for a in (x, mean, std)) or (std <= 0).any()):
        raise ValueError('Finite fitting-normalized causal features required')
    return np.sqrt(np.mean(((x-mean)/std)**2, axis=1))


def support_limit(distance, weights, quantile=.99):
    d, w = np.asarray(distance, float), np.asarray(weights, float)
    if (d.ndim != 1 or w.shape != d.shape or not np.isfinite(d).all()
            or not np.isfinite(w).all() or (d < 0).any() or (w < 0).any()
            or w.sum() <= 0 or quantile != .99):
        raise ValueError('Fixed .99 weighted fitting-source support required')
    order = np.argsort(d, kind='stable'); mass = np.cumsum(w[order])/w.sum()
    return float(d[order[min(np.searchsorted(mass, quantile), len(d)-1)]])


def static_guard(utility, easy_risk, moving, budget=.02):
    u, e = np.asarray(utility, float), np.asarray(easy_risk, float)
    m = np.asarray(moving)
    if (m.ndim != 1 or m.dtype != bool or u.shape != (len(m), 2) or e.shape != u.shape
            or not np.isfinite(u).all() or not np.isfinite(e).all()
            or (u < 0).any() or (e < 0).any() or budget != .02):
        raise ValueError('Past-only nonnegative utility/easy costs and fixed budget required')
    return m & (u[:, 0] > u[:, 1]) & (e[:, 1] <= budget*e[:, 0])


def decide(score, guard, supported, threshold):
    s, g, p = np.asarray(score, float), np.asarray(guard), np.asarray(supported)
    if (s.ndim != 1 or g.shape != s.shape or p.shape != s.shape or g.dtype != bool
            or p.dtype != bool or not np.isfinite(s).all()
            or (threshold is not None and (not np.isfinite(threshold) or threshold > 0))):
        raise ValueError('Finite causal scores, masks and nonpositive threshold required')
    return np.zeros(len(s), bool) if threshold is None else g & p & (s <= threshold)


def metrics(cv, error, take, easy_cut, hard_cut):
    cv, error, take = np.asarray(cv, float), np.asarray(error, float), np.asarray(take)
    if (cv.ndim != 1 or error.shape != cv.shape or take.shape != cv.shape or take.dtype != bool
            or not np.array_equal(np.isfinite(cv), np.isfinite(error))
            or np.isinf(cv).any() or np.isinf(error).any() or easy_cut <= 0 or hard_cut < easy_cut):
        raise ValueError('Paired known/unknown evaluation labels and causal actions required')
    known = np.isfinite(cv)
    if not known.any() or (cv[known] < 0).any() or (error[known] < 0).any():
        raise ValueError('Nonnegative known reference and candidate costs required')
    c, e, t = cv[known], error[known], take[known]
    selected = np.where(t, e, c); harm = np.maximum(e-c, 0)*t
    ratio = float(harm.sum()/c[t].sum()) if c[t].sum() > 0 else None
    result = dict(rows=int(known.sum()), unknown=int((~known).sum()), switches=int(t.sum()),
        switch_rate=float(t.mean()), selected_positive_harm_ratio=ratio,
        positive_harm_over_all_reference=float(harm.sum()/c.sum()) if c.sum() > 0 else None,
        zero_reference_harmed=int(((c == 0) & (harm > 0)).sum()), zero_reference_harm=float(harm[c == 0].sum()),
        selected_error_sum=float(selected.sum()), cv_error_sum=float(c.sum()),
        selected_positive_harm=float(harm.sum()), selected_reference=float(c[t].sum()))
    for name, mask in dict(all=np.ones(len(c), bool), easy=(c > 0)&(c <= easy_cut), hard=c >= hard_cut).items():
        result[name+'_gain_percent'] = float(100*(1-selected[mask].sum()/c[mask].sum())) if c[mask].sum() > 0 else None
    return result


def calibrate(score, guard, supported, cv, error, sites, *, calibration_sites,
              thresholds, easy_cut, hard_cut, min_selected=32):
    """Only two calibration sources may enter this label-reading function."""
    sites = np.asarray(sites)
    arrays = [np.asarray(a) for a in (score, guard, supported, cv, error)]
    grid = list(thresholds)
    if (len(set(calibration_sites)) != 2 or set(sites) != set(calibration_sites)
            or any(a.shape != sites.shape for a in arrays) or sites.ndim != 1
            or not grid or any(not np.isfinite(t) or t > 0 for t in grid)
            or grid != sorted(set(grid)) or min_selected < 1):
        raise ValueError('Exactly two declared calibration sources and fixed sorted grid required')
    candidates = []
    for threshold in grid:
        take = decide(score, guard, supported, threshold)
        per_site = {str(s): metrics(np.asarray(cv)[sites == s], np.asarray(error)[sites == s],
            take[sites == s], easy_cut, hard_cut) for s in sorted(set(sites))}
        valid = all(r['switches'] >= min_selected and r['selected_positive_harm_ratio'] is not None
            and r['selected_positive_harm_ratio'] <= .02 and r['zero_reference_harmed'] == 0
            and r['easy_gain_percent'] is not None and r['easy_gain_percent'] >= -2
            and r['all_gain_percent'] is not None and r['all_gain_percent'] > 0 for r in per_site.values())
        candidates.append(dict(threshold=threshold, eligible=bool(valid), sites=per_site))
    valid = [r for r in candidates if r['eligible']]
    if not valid:
        return dict(threshold=None, reason='no_supported_empirically_safe_threshold', candidates=candidates,
                    calibration_sites=sorted(calibration_sites), certificate=False)
    best = max(valid, key=lambda r: (np.mean([s['all_gain_percent'] for s in r['sites'].values()]),
        np.mean([s['switch_rate'] for s in r['sites'].values()]), -r['threshold']))
    return dict(threshold=best['threshold'], reason='calibration_only_best_mean_gain', candidates=candidates,
                calibration_sites=sorted(calibration_sites), certificate=False)
