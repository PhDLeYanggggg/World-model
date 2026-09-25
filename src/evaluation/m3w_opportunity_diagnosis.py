"""Conserved post-hoc error accounting, separate from causal gate reasons."""
import numpy as np

from src.world_model.m3w_european_conditional_risk import pointwise_rule

REASONS = ('source_support_abstain', 'no_observed_motion', 'nonpositive_utility',
           'zero_event_mass', 'risk_veto', 'switch')
CONTRIBUTIONS = ('oracle_gain', 'captured_gain', 'switched_harm', 'net_gain',
                 'oracle_regret', *('missed_'+r for r in REASONS[:-1]))


def causal_reasons(utility, moments, moving, support, *, budget=.02):
    u, m, motion, support = map(np.asarray, (utility, moments, moving, support))
    expected = pointwise_rule(u, m, motion, budget=budget, support_available=True)
    if support.dtype != bool or support.shape != u.shape:
        raise ValueError('Explicit per-row fitting-support Boolean mask required')
    result = np.full(len(u), 5, np.int8)
    pending = np.ones(len(u), bool)
    for code, fail in enumerate((~support, ~motion, u <= 0, m[:, 0] <= 0,
                                 m[:, 1] > budget*m[:, 0])):
        use = pending & fail
        result[use] = code
        pending &= ~fail
    np.testing.assert_array_equal(result == 5, expected & support)
    return result


def summarize_values(values, *, resamples=3000, seed=39271):
    if not values or any(v is None for v in values):
        return dict(equal_locality=None, ci95=None)
    v = np.asarray(values, float)
    if not np.isfinite(v).all():
        raise ValueError('Undefined values must be explicit, not NaN')
    ci = None
    if len(v) > 1 and resamples:
        draws = np.random.default_rng(seed).choice(v, size=(resamples, len(v)))
        ci = np.quantile(draws.mean(1), [.025, .975]).tolist()
    return dict(equal_locality=float(v.mean()), ci95=ci)


def _ratio(a, b):
    return float(a/b) if b > 0 else None


def opportunity_ledger(reference, candidate, reasons, scenes, *, expected_scenes,
                       subset=None, resamples=3000, seed=39271):
    r, c, why, sites = map(np.asarray, (reference, candidate, reasons, scenes))
    n = len(r)
    if (r.shape != (n,) or c.shape != r.shape or why.shape != r.shape or sites.shape != r.shape
            or not np.issubdtype(why.dtype, np.integer) or np.any((why < 0) | (why >= len(REASONS)))
            or not np.array_equal(np.isnan(r), np.isnan(c)) or np.isinf(r).any() or np.isinf(c).any()
            or np.any(r[np.isfinite(r)] < 0) or np.any(c[np.isfinite(c)] < 0)
            or len(set(expected_scenes)) != len(expected_scenes) or not expected_scenes
            or not set(sites).issubset(expected_scenes)):
        raise ValueError('Aligned nonnegative paired costs and fixed scene roster required')
    subset = np.ones(n, bool) if subset is None else np.asarray(subset)
    if subset.shape != (n,) or subset.dtype != bool:
        raise ValueError('Explicit Boolean diagnostic subset required')
    known = np.isfinite(r) & subset
    gain = np.where(known, np.maximum(r-c, 0), 0)
    harm = np.where(known, np.maximum(c-r, 0), 0)
    switched = why == 5
    rows = {}
    for site in expected_scenes:
        population = (sites == site) & subset
        use = population & known
        denominator = float(r[use].sum())
        gross = float(gain[use].sum())
        captured = float(gain[use & switched].sum())
        paid = float(harm[use & switched].sum())
        missed = {name: float(gain[use & (why == i)].sum()) for i, name in enumerate(REASONS[:-1])}
        np.testing.assert_allclose(gross, captured+sum(missed.values()), rtol=1e-12, atol=1e-10)
        actual = np.where(switched[use], c[use], r[use])
        np.testing.assert_allclose((r[use]-actual).sum(), captured-paid, rtol=1e-11, atol=1e-9)
        fields = dict(oracle_gain=gross, captured_gain=captured, switched_harm=paid,
                      net_gain=captured-paid, oracle_regret=gross-captured+paid,
                      **{'missed_'+k: v for k, v in missed.items()})
        rows[site] = dict(indexed_rows=int(population.sum()), supported_rows=int(use.sum()),
            unknown_rows=int((population & ~known).sum()),
            cv_error_sum=denominator, candidate_error_sum=float(c[use].sum()),
            gross_beneficial_rows=int((use & (gain > 0)).sum()),
            harmful_switched_rows=int((use & switched & (harm > 0)).sum()),
            counts={name: int((population & (why == i)).sum()) for i, name in enumerate(REASONS)},
            contributions_percent={k: _ratio(100*v, denominator) for k, v in fields.items()},
            gain_capture_fraction=_ratio(captured, gross))
    summary = {k: summarize_values([x['contributions_percent'][k] for x in rows.values()],
                   resamples=resamples, seed=seed) for k in CONTRIBUTIONS}
    oracle, capture = summary['oracle_gain']['equal_locality'], summary['captured_gain']['equal_locality']
    return dict(indexed_rows=int(subset.sum()), supported_rows=int(known.sum()),
                unknown_rows=int((subset & ~known).sum()), by_scene=rows, summary=summary,
                gain_capture_fraction=_ratio(capture, oracle) if oracle is not None else None,
                bootstrap_resamples=resamples, bootstrap_seed=seed,
                inference_policy=False, diagnostic_label_use=True)


def risk_bands(reference, candidate, utility, moments, reasons):
    r, c, u, m, why = map(np.asarray, (reference, candidate, utility, moments, reasons))
    known = np.isfinite(r) & np.isfinite(c)
    ratio = np.divide(m[:, 1], m[:, 0], out=np.full(len(m), np.inf), where=m[:, 0] > 0)
    groups = {'zero_mass': m[:, 0] <= 0}
    lower = -np.inf
    for upper, name in ((.02, 'le_002'), (.1, '002_to_01'), (.5, '01_to_05'),
                        (1., '05_to_1'), (np.inf, 'gt_1')):
        groups[name] = (m[:, 0] > 0) & (ratio > lower) & (ratio <= upper)
        lower = upper
    out = {}
    for name, mask in groups.items():
        use = mask & known
        positive = use & (u > 0)
        out[name] = dict(indexed_rows=int(mask.sum()), supported_rows=int(use.sum()),
            beneficial_rows=int((use & (c < r)).sum()), harmful_rows=int((use & (c > r)).sum()),
            utility_positive_rows=int(positive.sum()),
            utility_positive_beneficial_rows=int((positive & (c < r)).sum()),
            switched_rows=int((mask & (why == 5)).sum()))
    assert sum(v['indexed_rows'] for v in out.values()) == len(r)
    return out
