"""Offline incremental opportunity relative to a frozen protected-motion floor."""
import numpy as np

from src.evaluation.m3w_opportunity_diagnosis import REASONS, summarize_values

COMPONENTS = ('oracle_gain', 'captured_gain', 'missed_gain', 'selected_harm',
              'fallback_regression', 'fallback_relief', 'original_gain', 'rebased_gain',
              'rebase_advantage', 'signed_oracle_deficit', 'union_oracle_gain',
              *('missed_'+r for r in REASONS[:-1]))
SUBSETS = ('all', 'easy', 'hard', 'complete', 'partial', 'endpoint', 'easy_complete', 'hard_complete')


def annotation_subsets(valid, cv, *, easy_cut, hard_cut):
    valid, cv = np.asarray(valid), np.asarray(cv)
    if (valid.ndim != 2 or valid.dtype != bool or valid.shape[1] < 1 or cv.shape != (len(valid),)
            or not np.isfinite([easy_cut, hard_cut]).all() or not 0 <= easy_cut <= hard_cut
            or not np.array_equal(np.isnan(cv), valid.sum(1) == 0)):
        raise ValueError('Explicit future-label support and fixed fitting cutoffs required for evaluation only')
    complete = valid.all(1)
    easy, hard = (cv > 0) & (cv <= easy_cut), cv >= hard_cut
    return dict(all=np.ones(len(cv), bool), easy=easy, hard=hard, complete=complete,
                partial=valid.any(1) & ~complete, endpoint=valid[:, -1],
                easy_complete=easy & complete, hard_complete=hard & complete)


def floor_ledger(cv, floor, neural, reasons, scenes, *, expected_scenes, subset=None,
                 resamples=3000, seed=39271):
    cv, d, n = [np.asarray(v, float) for v in (cv, floor, neural)]
    why, sites = np.asarray(reasons), np.asarray(scenes)
    roster = tuple(expected_scenes)
    if (cv.ndim != 1 or any(v.shape != cv.shape for v in (d, n, why, sites))
            or not np.issubdtype(why.dtype, np.integer) or np.any((why < 0) | (why >= len(REASONS)))
            or any(not np.array_equal(np.isnan(v), np.isnan(cv)) for v in (d, n))
            or any(np.isinf(v).any() or (v[np.isfinite(v)] < 0).any() for v in (cv, d, n))
            or not roster or len(set(roster)) != len(roster) or not set(sites).issubset(roster)):
        raise ValueError('Aligned nonnegative costs, causal reasons and fixed locality roster required')
    subset = np.ones(len(cv), bool) if subset is None else np.asarray(subset)
    if subset.shape != cv.shape or subset.dtype != bool:
        raise ValueError('Explicit Boolean diagnostic subset required')
    switched = why == REASONS.index('switch')
    known = subset & np.isfinite(cv)
    benefit, harm = np.maximum(d-n, 0), np.maximum(n-d, 0)
    capture, missed = np.where(switched, benefit, 0), np.where(switched, 0, benefit)
    paid = np.where(switched, harm, 0)
    regression = np.where(switched, 0, np.maximum(cv-d, 0))
    relief = np.where(switched, 0, np.maximum(d-cv, 0))
    original = np.where(switched, n, cv)
    rebased = np.where(switched, n, d)
    oracle = np.minimum(d, n)
    union = np.minimum(cv, oracle)
    parts = dict(oracle_gain=benefit, captured_gain=capture, missed_gain=missed,
                 selected_harm=paid, fallback_regression=regression, fallback_relief=relief,
                 original_gain=capture-paid-regression+relief, rebased_gain=capture-paid,
                 rebase_advantage=regression-relief,
                 signed_oracle_deficit=missed+paid+regression-relief,
                 union_oracle_gain=d-union,
                 **{'missed_'+r: np.where(why == i, benefit, 0) for i, r in enumerate(REASONS[:-1])})
    for a, b in ((d-original, parts['original_gain']), (d-rebased, parts['rebased_gain']),
                 (original-oracle, parts['signed_oracle_deficit']), (original-rebased, parts['rebase_advantage']),
                 (benefit, capture+missed)):
        np.testing.assert_allclose(a[known], b[known], rtol=1e-11, atol=1e-9)
    rows = {}
    for site in roster:
        pop = subset & (sites == site)
        use = pop & known
        denominator = float(d[use].sum())
        sums = {k: float(v[use].sum()) for k, v in parts.items()}
        np.testing.assert_allclose(sums['missed_gain'], sum(sums['missed_'+r] for r in REASONS[:-1]),
                                   rtol=1e-11, atol=1e-9)
        rows[str(site)] = dict(indexed_rows=int(pop.sum()), supported_rows=int(use.sum()),
            unknown_rows=int((pop & ~known).sum()), switched_unknown_rows=int((pop & ~known & switched).sum()),
            floor_error_sum=denominator, original_error_sum=float(original[use].sum()),
            rebased_error_sum=float(rebased[use].sum()), oracle_error_sum=float(oracle[use].sum()),
            union_oracle_error_sum=float(union[use].sum()),
            counts={r: int((pop & (why == i)).sum()) for i, r in enumerate(REASONS)},
            sums=sums, contributions_percent={k: 100*v/denominator if denominator > 0 else None for k, v in sums.items()})
    summary = {k: summarize_values([v['contributions_percent'][k] for v in rows.values()],
                resamples=resamples, seed=seed) for k in COMPONENTS}
    return dict(indexed_rows=int(subset.sum()), supported_rows=int(known.sum()), unknown_rows=int((subset & ~known).sum()),
                by_scene=rows, summary=summary, reference='frozen_protected_damping_policy',
                diagnostic_label_use=True, deployable=False, future_labels_in_inference=False,
                attribution='ordered_gate_accounting_not_causal_mediation')
