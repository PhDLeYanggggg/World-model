"""Separate reference-cost and harm errors without outcome-based selection."""
import numpy as np
from src.evaluation.m3w_harm_tail_diagnostics import quantiles, top_mass_share


def split_controller(ids, sites, held_site, producer_sites, readout_sites):
    ids, sites = np.asarray(ids), np.asarray(sites)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or sites.shape != ids.shape
            or len(set(sites)) != 4 or held_site not in sites
            or set(sites) & (set(producer_sites) | set(readout_sites))):
        raise ValueError('Four disjoint controller localities required')
    held = sites == held_site
    return ids[~held], ids[held]


def labels(cv, error):
    cv, error = np.asarray(cv, float), np.asarray(error, float)
    if (cv.ndim != 1 or error.shape != cv.shape or np.isinf(cv).any() or np.isinf(error).any()
            or not np.array_equal(np.isnan(cv), np.isnan(error))
            or (cv[np.isfinite(cv)] < 0).any() or (error[np.isfinite(error)] < 0).any()):
        raise ValueError('Paired nonnegative cost labels required')
    return np.column_stack((cv, np.maximum(error-cv, 0)))


def screen(pred, risk=0.02):
    pred = np.asarray(pred, float)
    if pred.ndim != 2 or pred.shape[1] != 2 or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Finite nonnegative two-moment predictions required')
    if risk != 0.02:
        raise ValueError('Fixed diagnostic threshold; no policy search')
    return pred[:, 1] <= risk*pred[:, 0]


def score_edges(pred, weights, probabilities):
    screen(pred)
    return {name: quantiles(np.asarray(pred)[:, k], weights, probabilities)
            for k, name in enumerate(('reference', 'harm'))}


def ratio(a, b):
    return float(a/b) if b > 0 else None


def summarize(pred, target, constant, scale, edges):
    pred, target, constant = map(lambda x: np.asarray(x, float), (pred, target, constant))
    accepted = screen(pred)
    if target.shape != pred.shape or constant.shape != (2,) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Aligned labels and fitting-only constant/scale required')
    if np.isinf(target).any() or not np.array_equal(np.isnan(target[:, 0]), np.isnan(target[:, 1])):
        raise ValueError('Paired known labels required')
    known = np.isfinite(target).all(1)
    if not known.any():
        raise ValueError('Supported locality required')
    p, y, take = pred[known], target[known], accepted[known]
    w = np.full(len(y), 1/len(y))
    result = dict(rows=len(pred), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
                  screen_rate=float(take.mean()), screen_count=int(take.sum()),
                  zero_reference_rows=int((y[:, 0] == 0).sum()),
                  screen_zero_reference_harmed=int((take & (y[:, 0] == 0) & (y[:, 1] > 0)).sum()),
                  screen_positive_harm_mean=float(y[take, 1].mean()) if take.any() else None)
    for k, name in enumerate(('reference', 'harm')):
        mse = float(np.mean((p[:, k]-y[:, k])**2))
        const_mse = float(np.mean((constant[k]-y[:, k])**2))
        result.update({name+'_MSE_scaled': mse/scale**2,
            name+'_constant_MSE_scaled': const_mse/scale**2,
            name+'_MSE_skill_percent': 100*(1-mse/const_mse) if const_mse > 0 else None,
            name+'_actual_mean': float(y[:, k].mean()), name+'_predicted_mean': float(p[:, k].mean()),
            name+'_actual_over_predicted': ratio(y[:, k].mean(), p[:, k].mean())})
        result['screen_'+name+'_actual_over_predicted'] = ratio(y[take, k].mean(), p[take, k].mean()) if take.any() else None
        result['screen_'+name+'_actual_mean'] = float(y[take, k].mean()) if take.any() else None
        result['screen_'+name+'_predicted_mean'] = float(p[take, k].mean()) if take.any() else None
        ix = np.searchsorted(edges[name], p[:, k], side='left')
        result[name+'_bins'] = [dict(index=b, rows=int((ix == b).sum()),
            actual_sum=float(y[ix == b, k].sum()), predicted_sum=float(p[ix == b, k].sum()))
            for b in range(len(edges[name])+1)]
    result['screen_actual_harm_ratio'] = ratio(y[take, 1].sum(), y[take, 0].sum()) if take.any() else None
    result['screen_predicted_harm_ratio'] = ratio(p[take, 1].sum(), p[take, 0].sum()) if take.any() else None
    result['top10_harm_mass_share'] = top_mass_share(p[:, 1], y[:, 1], w, .1)
    result['oracle_top10_harm_mass_share'] = top_mass_share(y[:, 1], y[:, 1], w, .1)
    return result


MEASURES = ['reference_MSE_skill_percent', 'harm_MSE_skill_percent',
    'reference_actual_over_predicted', 'harm_actual_over_predicted', 'screen_rate',
    'screen_reference_actual_over_predicted', 'screen_harm_actual_over_predicted',
    'screen_actual_harm_ratio', 'screen_predicted_harm_ratio', 'top10_harm_mass_share']


def flatten_pair(held, fit):
    if len(fit) != 3:
        raise ValueError('Exactly three fitting-locality summaries required')
    out = {}
    for key in MEASURES:
        values = [v[key] for v in fit]
        mean = float(np.mean(values)) if all(v is not None for v in values) else None
        out['held_'+key] = held[key]
        out['fit_'+key] = mean
        out['gap_'+key] = held[key]-mean if held[key] is not None and mean is not None else None
    return out
