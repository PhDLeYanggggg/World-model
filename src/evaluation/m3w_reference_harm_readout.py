"""Paired forecast diagnostics, retaining missing and zero-reference support."""
import numpy as np


def harm_summary(model, reference, mask=None):
    m, r = np.asarray(model, float), np.asarray(reference, float)
    use = np.ones(len(m), bool) if mask is None else np.asarray(mask)
    if (m.ndim != 1 or r.shape != m.shape or use.shape != m.shape or use.dtype != bool
            or np.isinf(m).any() or np.isinf(r).any()
            or not np.array_equal(np.isnan(m), np.isnan(r))
            or np.any(m[np.isfinite(m)] < 0) or np.any(r[np.isfinite(r)] < 0)):
        raise ValueError('Aligned nonnegative errors and identical label support required')
    supported = use & np.isfinite(m)
    rr, mm = r[supported], m[supported]
    denom = float(rr.sum())
    positive = float(np.maximum(mm-rr, 0).sum())
    benefit = float(np.maximum(rr-mm, 0).sum())
    ratio = 100*positive/denom if denom > 0 else None
    net = 100*(float(mm.sum())-denom)/denom if denom > 0 else None
    return dict(indexed_rows=int(use.sum()), supported_rows=int(supported.sum()),
        unknown_rows=int((use & ~np.isfinite(m)).sum()), reference_error_sum=denom,
        positive_harm_sum=positive, positive_benefit_sum=benefit,
        known_positive_harm_percent=ratio, known_net_degradation_percent=net,
        known_budget_pass=None if ratio is None else ratio <= 2.,
        complete_population_budget_claim=False,
        support_status='not_estimable' if ratio is None else 'supported_labels_only',
        zero_reference_rows=int((rr == 0).sum()),
        zero_reference_harmed_rows=int(((rr == 0) & (mm > 0)).sum()),
        zero_reference_added_error=float(mm[rr == 0].sum()),
        model_p95=None if not len(mm) else float(np.quantile(mm, .95)),
        model_p99=None if not len(mm) else float(np.quantile(mm, .99)))


def compare_arms(new, old, reference, sites, roster, resamples=3000, seed=39271):
    """Improvement difference in reference-relative pp, resampling localities."""
    new, old, ref, sites = map(np.asarray, (new, old, reference, sites))
    if new.shape != old.shape or new.shape != ref.shape or new.shape != sites.shape:
        raise ValueError('Paired aligned arrays required')
    if not np.array_equal(np.isnan(new), np.isnan(old)) or not np.array_equal(np.isnan(new), np.isnan(ref)):
        raise ValueError('Identical outcome support required')
    effects = {}
    for site in roster:
        use = (sites == site) & np.isfinite(ref)
        denom = ref[use].sum()
        effects[str(site)] = None if not use.any() or denom <= 0 else float(100*(old[use]-new[use]).sum()/denom)
    defined = all(v is not None for v in effects.values())
    values = list(effects.values())
    ci = None
    if defined and len(values) > 1 and resamples:
        draws = np.random.default_rng(seed).choice(values, size=(resamples, len(values)))
        ci = np.quantile(draws.mean(1), [.025, .975]).tolist()
    return dict(by_locality_pp=effects, equal_locality_gain_difference_pp=float(np.mean(values)) if defined else None,
        paired_locality_bootstrap_ci95_pp=ci, bootstrap_resamples=resamples, bootstrap_seed=seed,
        uncertainty='conditional_source_development_not_independent_confirmation',
        seed_average='errors_not_ensemble_predictions')
