"""Frozen-action diagnostics; none of these functions deploys a policy."""
import numpy as np
from scipy.spatial import cKDTree


def signed_cost(cv, floor, neural, easy_cut, scale):
    cv, floor, neural = map(lambda a: np.asarray(a, float), (cv, floor, neural))
    known = np.isfinite(cv)
    if (cv.ndim != 1 or floor.shape != cv.shape or neural.shape != cv.shape
            or not np.array_equal(known, np.isfinite(floor))
            or not np.array_equal(known, np.isfinite(neural))
            or any(np.isinf(a).any() or (a[known] < 0).any() for a in (cv, floor, neural))
            or scale <= 0 or easy_cut <= 0):
        raise ValueError('Aligned nonnegative costs, unknown support and positive units required')
    harm = np.maximum(neural-floor, 0.)
    excess = (harm-.02*floor)/scale
    easy = known & (cv > 0) & (cv <= easy_cut)
    out = np.column_stack((excess, np.where(easy, excess, 0.)))
    out[~known] = np.nan
    return out


def query_sizes(sites, recordings, frames):
    keys = list(zip(map(str, sites), map(str, recordings), map(int, frames)))
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0)+1
    return np.array([counts[k] for k in keys], dtype=np.int64)


def support_from_fit(u_fit, fit_sites, known_fit, u_eval, mean, std):
    """Two-source nearest-descriptor distances; no evaluation outcome argument.

    A source's radius is the 95th percentile of distances from the other
    fitting source. Temporal duplicates are not independent observations.
    This is a covariate support descriptor, not epistemic uncertainty.
    """
    u_fit, u_eval, mean, std = map(lambda a: np.asarray(a, float), (u_fit, u_eval, mean, std))
    fit_sites, known_fit = np.asarray(fit_sites), np.asarray(known_fit)
    if (u_fit.ndim != 2 or u_eval.ndim != 2 or u_fit.shape[1] != 6 or u_eval.shape[1] != 6
            or mean.shape != (6,) or std.shape != (6,) or (std <= 0).any()
            or known_fit.shape != (len(u_fit),) or known_fit.dtype != bool
            or fit_sites.shape != (len(u_fit),) or len(set(fit_sites)) != 2
            or not all(np.isfinite(x).all() for x in (u_fit, u_eval, mean, std))):
        raise ValueError('Two fitting sources and six finite causal descriptors required')
    train = (u_fit-mean)/std
    evaluate = (u_eval-mean)/std
    distances, thresholds, names = [], [], sorted(set(fit_sites))
    for site in names:
        own = known_fit & (fit_sites == site)
        other = known_fit & (fit_sites != site)
        if not own.any() or not other.any():
            raise ValueError('Known fitting rows in each source required')
        tree = cKDTree(train[own])
        radius = float(np.quantile(tree.query(train[other], workers=1)[0], .95))
        distances.append(tree.query(evaluate, workers=1)[0])
        thresholds.append(radius)
    d = np.column_stack(distances)
    outside = d > np.array(thresholds)[None, :]
    return dict(distance=d, thresholds=thresholds, fitting_sites=list(map(str, names)),
                outside_both=outside.all(1), outside_either=outside.any(1))


def residual_metrics(prediction, truth, selected, bank, support, size):
    prediction, truth = map(lambda a: np.asarray(a, float), (prediction, truth))
    selected, bank, support, size = map(np.asarray, (selected, bank, support, size))
    n = len(truth)
    if (truth.shape != (n, 2) or prediction.shape != truth.shape
            or not np.isfinite(prediction).all() or np.isinf(truth).any()
            or not np.array_equal(np.isfinite(truth[:, 0]), np.isfinite(truth[:, 1]))
            or selected.shape != (n,) or selected.dtype != bool
            or bank.shape != (n, 3) or bank.dtype != bool
            or support.shape != (n,) or support.dtype != bool or size.shape != (n,)):
        raise ValueError('Aligned scores, known outcomes and frozen Boolean masks required')
    use = selected & np.isfinite(truth).all(1)
    count = int(use.sum())
    out = dict(rows=int(selected.sum()), known=count, unknown=int(selected.sum()-count))
    for j, axis in enumerate(('all', 'easy')):
        error = truth[use, j]-prediction[use, j]
        out[axis+'_optimism_mean'] = float(error.mean()) if count else None
        out[axis+'_MSE'] = float(np.mean(error**2)) if count else None
        out[axis+'_actual_positive_fraction'] = float(np.mean(truth[use, j] > 0)) if count else None
        out[axis+'_predicted_excess_sum'] = float(prediction[use, j].sum())
        out[axis+'_actual_excess_sum'] = float(truth[use, j].sum())
    for name, mask in [('controller_proxy', bank[:, 0]), ('low_proxy', bank[:, 1]),
                       ('high_proxy', bank[:, 2]), ('outside_both_fit_sources', support),
                       ('singleton_query', size == 1)]:
        out[name+'_fraction'] = float(mask[selected].mean()) if selected.any() else None
        out[name+'_known_fraction'] = float(mask[use].mean()) if count else None
    return out


def query_residuals(prediction, truth, take, recordings, frames):
    """Selected sums within each frozen locality/recording/frame query."""
    prediction, truth, take = np.asarray(prediction), np.asarray(truth), np.asarray(take)
    groups = {}
    for i in range(len(take)):
        groups.setdefault((str(recordings[i]), int(frames[i])), []).append(i)
    records = []
    for positions in groups.values():
        at = np.asarray(positions, int)
        sel = at[take[at]]
        known = sel[np.isfinite(truth[sel]).all(1)]
        if not len(sel):
            continue
        records.append((len(sel), len(known), prediction[sel].sum(0),
                        truth[known].sum(0), prediction[known].sum(0)))
    complete = [r for r in records if r[0] == r[1]]
    out = dict(selected_queries=len(records), complete_queries=len(complete),
               unknown_selected_queries=len(records)-len(complete))
    for j, key in enumerate(('all', 'easy')):
        out[key+'_predicted_safe_actual_excess_queries'] = int(sum(r[2][j] <= 1e-10 and r[3][j] > 1e-10 for r in complete))
        out[key+'_optimism_mean_complete_query'] = float(np.mean([r[3][j]-r[2][j] for r in complete])) if complete else None
    return out


def exchange_accounting(floor, neural, before, after):
    floor, neural = map(lambda a: np.asarray(a, float), (floor, neural))
    before, after = np.asarray(before), np.asarray(after)
    if (before.dtype != bool or after.dtype != bool or floor.shape != neural.shape
            or before.shape != floor.shape or after.shape != floor.shape
            or not np.array_equal(np.isfinite(floor), np.isfinite(neural))):
        raise ValueError('Aligned costs and frozen actions required')
    known = np.isfinite(floor)
    den = float(floor[known].sum())
    benefit, harm = np.maximum(floor-neural, 0), np.maximum(neural-floor, 0)
    out = dict(floor_sum=den, selected_before=int(before.sum()), selected_after=int(after.sum()))
    for name, mask in [('added', after & ~before), ('removed', before & ~after), ('shared', before & after)]:
        use = known & mask
        out[name+'_count'] = int(mask.sum())
        out[name+'_unknown'] = int((mask & ~known).sum())
        out[name+'_benefit'] = float(benefit[use].sum())
        out[name+'_harm'] = float(harm[use].sum())
        for key in ('benefit', 'harm'):
            out[name+'_'+key+'_over_floor_pp'] = 100*out[name+'_'+key]/den if den > 0 else None
    out['gain_change'] = out['added_benefit']-out['added_harm']-out['removed_benefit']+out['removed_harm']
    out['gain_change_over_floor_pp'] = 100*out['gain_change']/den if den > 0 else None
    direct = float((np.where(before, neural, floor)[known]-np.where(after, neural, floor)[known]).sum())
    np.testing.assert_allclose(out['gain_change'], direct, rtol=1e-10, atol=1e-8)
    return out
