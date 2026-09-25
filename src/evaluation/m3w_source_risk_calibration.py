"""Empirical source-only calibration, not conformal risk certification."""
import numpy as np

from src.world_model.m3w_european_conditional_risk import event_labels, pointwise_rule


def apply_calibration(utility, moments, moving, rule, budget=.02):
    # Validate causal arrays even when the fitted rule requests abstention.
    raw = pointwise_rule(utility, moments, moving, budget=budget, support_available=True)
    if rule['abstain']:
        return np.zeros(len(raw), bool)
    if rule['kind'] == 'none':
        return raw
    if rule['kind'] == 'selected_risk_grid':
        tau = rule['threshold']
        if not np.isfinite(tau) or not 0 <= tau <= budget:
            raise ValueError('Calibration cannot relax the registered ratio budget')
        return pointwise_rule(utility, moments, moving, budget=tau, support_available=True)
    if rule['kind'] == 'population_rescale':
        h, d = rule['harm_multiplier'], rule['denominator_multiplier']
        if not np.isfinite([h, d]).all() or h < 1 or not 0 < d <= 1:
            raise ValueError('Only conservative moment rescaling is allowed')
        adjusted = np.asarray(moments)*np.array([d, h])
        return pointwise_rule(utility, adjusted, moving, budget=budget, support_available=True)
    raise ValueError('Unknown frozen calibration rule')


def calibration_evidence(bits, reference, candidate, sites, easy_cut, budget):
    r, p, s = np.asarray(reference), np.asarray(candidate), np.asarray(sites)
    if (bits.dtype != bool or bits.shape != r.shape or p.shape != r.shape or s.shape != r.shape
            or np.isinf(r).any() or np.isinf(p).any() or np.any(r[np.isfinite(r)] < 0)
            or np.any(p[np.isfinite(p)] < 0) or not np.array_equal(np.isnan(r), np.isnan(p))):
        raise ValueError('Aligned supported nonnegative calibration labels required')
    known = np.isfinite(r)
    rows, gains, feasible = {}, [], True
    for site in sorted(set(s)):
        use = (s == site) & known
        easy = use & (r > 0) & (r <= easy_cut)
        zero = use & (r == 0)
        if not use.any() or r[use].sum() <= 0:
            raise ValueError('Each calibration locality needs positive CV mass')
        added = np.where(bits[use], p[use]-r[use], 0)
        easy_added = np.where(bits[easy], p[easy]-r[easy], 0)
        harm, mass = float(np.maximum(added, 0).sum()), float(r[use].sum())
        eh, em = float(np.maximum(easy_added, 0).sum()), float(r[easy].sum())
        zero_harm = int((bits[zero] & (p[zero] > 0)).sum())
        good = harm <= budget*mass and eh <= budget*em and zero_harm == 0
        gain = float(-100*added.sum()/mass)
        rows[str(site)] = dict(rows=int(use.sum()), selected=int(bits[use].sum()),
            positive_harm=harm, cv_mass=mass, positive_easy_harm=eh, easy_cv_mass=em,
            positive_harm_ratio=harm/mass, positive_easy_harm_ratio=eh/em if em > 0 else None,
            zero_harmed=zero_harm, net_gain_percent=gain, feasible=bool(good))
        gains.append(gain)
        feasible &= good
    return dict(by_locality=rows, feasible=bool(feasible), equal_locality_gain_percent=float(np.mean(gains)),
        selected=int(bits.sum()), selected_unknown=int((bits & ~known).sum()),
        independently_calibrated=False)


def fit_calibration(utility, moments, moving, reference, candidate, sites, *, easy_cut,
                    event, grid, budget=.02):
    if sorted(set(grid)) != list(grid) or grid[0] != 0 or grid[-1] != budget:
        raise ValueError('Complete ordered preregistered calibration grid required')
    raw = dict(kind='none', abstain=False)
    base = apply_calibration(utility, moments, moving, raw, budget)
    baseline_evidence = calibration_evidence(base, reference, candidate, sites, easy_cut, budget)
    truth = event_labels(reference, np.maximum(candidate-reference, 0), easy_cut=easy_cut, event=event)
    known = np.isfinite(truth).all(1)
    predicted_means, true_means = [], []
    for site in sorted(set(sites)):
        use = (sites == site) & known
        predicted_means.append(np.asarray(moments)[use].mean(0))
        true_means.append(truth[use].mean(0))
    ph, th = np.mean(predicted_means, 0), np.mean(true_means, 0)
    unsupported = ph[0] <= 0 or th[0] <= 0 or (ph[1] <= 0 and th[1] > 0)
    rescale = dict(kind='population_rescale', abstain=bool(unsupported),
        harm_multiplier=max(1., float(th[1]/ph[1])) if ph[1] > 0 else 1.,
        denominator_multiplier=min(1., float(th[0]/ph[0])) if ph[0] > 0 and th[0] > 0 else 1.)
    records = []
    fallback = dict(kind='selected_risk_grid', abstain=True, threshold=None)
    best, best_rank = fallback, (0., 0, 0.)
    for tau in grid:
        rule = dict(kind='selected_risk_grid', abstain=False, threshold=float(tau))
        bits = apply_calibration(utility, moments, moving, rule, budget)
        e = calibration_evidence(bits, reference, candidate, sites, easy_cut, budget)
        records.append(dict(threshold=float(tau), **e))
        rank = (e['equal_locality_gain_percent'], -e['selected'], -float(tau))
        if e['feasible'] and rank > best_rank:
            best, best_rank = rule, rank
    return dict(rules=dict(none=raw, population_rescale=rescale, selected_risk_grid=best),
        calibration_localities=sorted(set(sites)), raw_evidence=baseline_evidence,
        grid_evidence=records, calibration_predicted_mean=ph.tolist(), calibration_true_mean=th.tolist(),
        calibrated_guarantee=False, selection_data_role='source_internal_calibration_only')
