"""Fixed-model source-gap diagnostics, never a deployment or threshold fitter."""
from collections import defaultdict
import numpy as np


def source_weights(sites):
    sites = np.asarray(sites)
    if sites.ndim != 1 or not len(sites):
        raise ValueError('Nonempty fitting sources required')
    w = np.zeros(len(sites), float)
    for s in np.unique(sites):
        at = sites == s
        w[at] = 1 / (len(np.unique(sites)) * at.sum())
    return w


def fit_edges(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if (v.ndim != 1 or w.shape != v.shape or not len(v) or
            not np.isfinite(v).all() or not np.isfinite(w).all() or
            (w < 0).any() or not np.isclose(w.sum(), 1)):
        raise ValueError('Finite fitting values and normalized weights required')
    order = np.argsort(v, kind='stable')
    cdf = np.cumsum(w[order])
    return v[order[np.minimum(np.searchsorted(cdf, [.25, .75]), len(v)-1)]]


def assign_bins(values, edges):
    v, e = np.asarray(values, float), np.asarray(edges, float)
    if e.shape != (2,) or np.any(np.diff(e) < 0) or not np.isfinite(e).all() or not np.isfinite(v).all():
        raise ValueError('Finite values and ordered fitting quartiles required')
    return np.searchsorted(e, v, side='right')


def causal_axes(geometry, floor, candidate, x, preprocess, support_limit):
    g, b, p, x = (np.asarray(a, float) for a in (geometry, floor, candidate, x))
    n = len(g)
    if (g.shape != (n, 476) or b.shape != (n, 12, 2) or p.shape != b.shape or
            x.shape != (n, 380) or support_limit <= 0 or
            not all(np.isfinite(a).all() for a in (g, b, p, x))):
        raise ValueError('Frozen causal schema required')
    h = g[:, :16].reshape(n, 8, 2)
    t = g[:, 16:24]
    mask = g[:, 230:294].reshape(n, 8, 8)
    nt = g[:, 166:230].reshape(n, 8, 8)
    if (not np.isin(mask, [0, 1]).all() or np.any(t > 0) or
            np.any(np.diff(t, axis=1) <= 0) or np.any(t[:, -1] != 0) or
            np.any(nt[mask.astype(bool)] > 0)):
        raise ValueError('Only valid current/past timestamps allowed')
    nb = g[:, 38:166].reshape(n, 8, 8, 2)
    extent = np.maximum(1, np.maximum(np.linalg.norm(h, axis=-1).max(1),
        np.where(mask > 0, np.linalg.norm(nb, axis=-1), 0).max((1, 2))))
    step = np.diff(h, axis=1)
    speed = np.linalg.norm(step, axis=-1)
    path = speed.sum(1)
    nonlinearity = 1 - np.divide(np.linalg.norm(h[:, -1]-h[:, 0], axis=1), path,
        out=np.ones(n), where=path > 0)
    prod = speed[:, 1:] * speed[:, :-1]
    cosine = np.divide((step[:, 1:]*step[:, :-1]).sum(-1), prod,
        out=np.ones_like(prod), where=prod > 0)
    angles = np.arccos(np.clip(cosine, -1, 1))
    turn = np.divide(angles.sum(1), (prod > 0).sum(1), out=np.zeros(n), where=(prod > 0).sum(1) > 0)
    z = (x-preprocess['mean'])/preprocess['std']
    return dict(mean_step_over_extent=speed.mean(1)/extent,
        last_step_over_extent=speed[:, -1]/extent,
        path_nonlinearity=np.clip(nonlinearity, 0, 1), mean_turn_radians=turn,
        neighbor_occupancy=mask.mean((1, 2)),
        rollout_disagreement_over_extent=np.linalg.norm(p-b, axis=-1).mean(1)/extent,
        feature_radius_over_limit=np.sqrt(np.mean(z*z, axis=1))/support_limit,
        clipped_feature_fraction=(np.abs(z) > preprocess['clip']).mean(1))


def evaluate_slice(mask, floor, neural, take, eligible, predicted_excess, scale):
    mask, take, eligible = [np.asarray(a) for a in (mask, take, eligible)]
    f, n, q = [np.asarray(a, float) for a in (floor, neural, predicted_excess)]
    if (any(a.shape != f.shape for a in (mask, take, eligible, n, q)) or f.ndim != 1 or
            any(a.dtype != bool for a in (mask, take, eligible)) or (take & ~eligible).any() or
            not np.array_equal(np.isfinite(f), np.isfinite(n)) or np.isinf(f).any() or np.isinf(n).any() or
            not np.isfinite(q).all() or not np.isfinite(scale) or scale <= 0 or
            (f[np.isfinite(f)] < 0).any() or (n[np.isfinite(n)] < 0).any()):
        raise ValueError('Aligned frozen decisions and separately supplied evaluation labels required')
    known = mask & np.isfinite(f)
    selected = known & take
    supported = known & eligible
    fn, nn = f/scale, n/scale
    benefit, harm = np.maximum(fn-nn, 0), np.maximum(nn-fn, 0)
    target = harm-.02*fn
    error = q-target
    out = dict(rows=int(mask.sum()), known=int(known.sum()), selected=int((mask & take).sum()),
        selected_known=int(selected.sum()), unknown_selected=int((mask & take & ~np.isfinite(f)).sum()),
        eligible=int((mask & eligible).sum()), eligible_known=int(supported.sum()),
        floor_sum=float(fn[known].sum()), selected_reference_sum=float(fn[selected].sum()),
        benefit_sum=float(benefit[selected].sum()), harm_sum=float(harm[selected].sum()),
        oracle_benefit_sum=float(benefit[known].sum()),
        eligible_oracle_benefit_sum=float(benefit[supported].sum()),
        harmful_selected=int((selected & (harm > 0)).sum()),
        over_budget_selected=int((selected & (target > 0)).sum()),
        sq_error_sum=float((error[known]**2).sum()), bias_sum=float(error[known].sum()),
        eligible_sq_error_sum=float((error[supported]**2).sum()),
        selected_sq_error_sum=float((error[selected]**2).sum()),
        selected_predicted_excess_sum=float(q[selected].sum()),
        selected_observed_excess_sum=float(target[selected].sum()))
    return out


def derived(s):
    def ratio(a, b, multiplier=1.):
        return multiplier*a/b if b > 0 else None
    return dict(known_percent=ratio(s['known'], s['rows'], 100),
        intervention_percent=ratio(s['selected'], s['rows'], 100),
        eligible_percent=ratio(s['eligible'], s['rows'], 100),
        unknown_selected_percent=ratio(s['unknown_selected'], s['selected'], 100),
        harm_percent=ratio(s['harm_sum'], s['selected_reference_sum'], 100),
        net_gain_percent=ratio(s['benefit_sum']-s['harm_sum'], s['floor_sum'], 100),
        oracle_gain_percent=ratio(s['oracle_benefit_sum'], s['floor_sum'], 100),
        eligible_oracle_gain_percent=ratio(s['eligible_oracle_benefit_sum'], s['floor_sum'], 100),
        captured_benefit_percent=ratio(s['benefit_sum'], s['oracle_benefit_sum'], 100),
        harmful_selected_percent=ratio(s['harmful_selected'], s['selected_known'], 100),
        score_MSE=ratio(s['sq_error_sum'], s['known']),
        score_bias=ratio(s['bias_sum'], s['known']),
        eligible_score_MSE=ratio(s['eligible_sq_error_sum'], s['eligible_known']),
        selected_score_MSE=ratio(s['selected_sq_error_sum'], s['selected_known']),
        selected_predicted_excess=ratio(s['selected_predicted_excess_sum'], s['selected_known']),
        selected_observed_excess=ratio(s['selected_observed_excess_sum'], s['selected_known']))


def reduce_slices(rows, sites, resamples=3000, seed=101531):
    # Aggregate dependent seed/fit views within each fixed locality before bootstrap.
    groups = defaultdict(dict)
    for r in rows:
        key = (r['role'], r['policy'], r['axis']+'/'+r['bin'])
        dest = groups[key].setdefault(r['site'], defaultdict(float))
        for k, v in r['sums'].items(): dest[k] += v
    result = {}
    draws = np.random.default_rng(seed).integers(0, len(sites), (resamples, len(sites)))
    for (role, policy, label), values in groups.items():
        local = {s: derived(v) for s, v in values.items()}
        out = dict(sums_by_site={s: dict(v) for s, v in values.items()},
            dependent_views=sum(r['role']==role and r['policy']==policy and r['axis']+'/'+r['bin']==label for r in rows))
        for k in next(iter(local.values())):
            d = {s: local.get(s, {}).get(k) for s in sites}
            good = all(v is not None for v in d.values())
            a = np.array(list(d.values()), float)
            out[k] = dict(point=float(a.mean()) if good else None,
                ci95=np.quantile(a[draws].mean(1), [.025, .975]).tolist() if good else None,
                by_site=d, defined_localities=sum(v is not None for v in d.values()))
        result.setdefault(role, {}).setdefault(policy, {})[label] = out
    return result
