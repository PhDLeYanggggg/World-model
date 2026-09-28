"""Fitting-only utility/risk accounting. Oracle rankings never produce actions."""
import numpy as np
from src.world_model.m3w_easy_component_diagnostic import weights

REASONS = ('not_moving', 'unsupported', 'nonpositive_utility', 'all_risk_rejected',
           'easy_risk_rejected', 'admitted')
BUDGET = .02


def causal_screens(utility, moving, supported, risks):
    utility, moving, supported = map(np.asarray, (utility, moving, supported))
    n = len(utility)
    if (utility.shape != (n,) or not np.isfinite(utility).all() or moving.shape != (n,)
            or supported.shape != (n,) or moving.dtype != bool or supported.dtype != bool):
        raise ValueError('Finite utility and aligned causal guards required')
    result = {}
    for arm, value in risks.items():
        q = np.asarray(value)
        if q.shape != (n, 2) or not np.isfinite(q).all():
            raise ValueError('Two finite signed risk scores required')
        masks = {}; remaining = np.ones(n, bool)
        failures = (~moving, ~supported, utility <= 0, q[:, 0] > 0, q[:, 1] > 0)
        for name, failure in zip(REASONS[:-1], failures):
            masks[name] = remaining & failure
            remaining = remaining & ~failure
        masks['admitted'] = remaining; result[arm] = masks
    return result


def ratio(a, b):
    return float(a/b) if b > 0 else None


def _readout(ids, sites, recordings, frames, utility, moving, supported, risks,
             known, easy, reference, benefit, harm):
    screens = causal_screens(utility, moving, supported, risks)
    w, _ = weights(sites, recordings, frames, known)
    e, r, b, h = (np.where(known, v, 0.) for v in (easy, reference, benefit, harm))
    gain = b-h; qtrue = np.column_stack((h-BUDGET*r, e*(h-BUDGET*r)))
    keys = np.rec.fromarrays([sites.astype(str), recordings.astype(str), frames], names='site,recording,frame')
    _, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(inv, kind='stable'); cuts = np.flatnonzero(np.diff(inv[order]))+1
    queries = np.split(order, cuts)
    eligible = moving & supported & (utility > 0)
    out = dict(rows=len(ids), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        query_count=len(queries), total=dict(benefit=float(w@b), harm=float(w@h),
            net_gain=float(w@gain), reference=float(w@r)),
        utility=dict(weighted_MSE=float(w@((utility-gain)**2)),
            weighted_bias=float(w@(utility-gain)),
            positive_score_harm_mass=float(w@(h*(utility > 0))),
            nonpositive_score_benefit_mass=float(w@(b*(utility <= 0)))), arms={})
    for arm, masks in screens.items():
        selected = masks['admitted']; q = risks[arm]
        partition = {name: dict(rows=int(mask.sum()), known_rows=int((known & mask).sum()),
            mass=float(w@mask), benefit=float(w@(b*mask)), harm=float(w@(h*mask)))
            for name, mask in masks.items()}
        np.testing.assert_allclose(sum(p['benefit'] for p in partition.values()), out['total']['benefit'], atol=1e-12)
        risk_stats = {key: dict(queries=len(queries), defined=0, violating=0, undefined=0,
            unknown_selected=0, predicted_safe_realized_positive=0,
            selected_reference_sum=0., selected_harm_sum=0.) for key in ('all', 'easy')}
        ranks = dict(queries=len(queries), queries_incomplete=0, queries_zero_count=0,
            queries_no_choice=0, queries_informative=0, oracle_minus_utility_sum=0.,
            oracle_minus_screen_sum=0., utility_minus_screen_sum=0.,
            oracle_better_than_utility_queries=0, utility_worse_than_screen_queries=0)
        for at in queries:
            take = at[selected[at]]
            for col, key in enumerate(('all', 'easy')):
                s = risk_stats[key]
                if (~known[take]).any(): s['unknown_selected'] += 1; continue
                multiplier = np.ones(len(take)) if key == 'all' else e[take]
                den = float((r[take]*multiplier).sum()); num = float((h[take]*multiplier).sum())
                s['selected_reference_sum'] += den; s['selected_harm_sum'] += num
                if den <= 0: s['undefined'] += 1
                else: s['defined'] += 1; s['violating'] += int(num > BUDGET*den)
                s['predicted_safe_realized_positive'] += int(q[take, col].sum() <= 0 and qtrue[take, col].sum() > 0)
            if not known[at].all(): ranks['queries_incomplete'] += 1; continue
            k = len(take); pool = at[eligible[at]]
            if k == 0: ranks['queries_zero_count'] += 1; continue
            if len(pool) <= k: ranks['queries_no_choice'] += 1; continue
            ranks['queries_informative'] += 1
            pred = pool[np.lexsort((ids[pool], -utility[pool]))[:k]]
            oracle = pool[np.lexsort((ids[pool], -gain[pool]))[:k]]
            actual, predicted, upper = (float(gain[x].sum()) for x in (take, pred, oracle))
            if upper < max(actual, predicted)-1e-9: raise AssertionError('Realized oracle upper bound violated')
            ranks['oracle_minus_utility_sum'] += upper-predicted
            ranks['oracle_minus_screen_sum'] += upper-actual
            ranks['utility_minus_screen_sum'] += predicted-actual
            ranks['oracle_better_than_utility_queries'] += int(upper > predicted+1e-12)
            ranks['utility_worse_than_screen_queries'] += int(predicted < actual-1e-12)
        # These are observed row/query errors on fitting labels, not calibrated risks.
        row = dict(selected_rows=int(selected.sum()), unknown_selected_rows=int((selected & ~known).sum()),
            mass=float(w@selected), benefit=float(w@(b*selected)), harm=float(w@(h*selected)),
            net_gain=float(w@(gain*selected)),
            selected_positive_harm_ratio=ratio(w@(h*selected), w@(r*selected)),
            easy_selected_positive_harm_ratio=ratio(w@(e*h*selected), w@(e*r*selected)),
            false_safe_all_mass=float(w@(selected & (qtrue[:, 0] > 0))),
            false_safe_easy_mass=float(w@(selected & (qtrue[:, 1] > 0))),
            easy_positive_harm=float(w@(e*h*selected)),
            risk_MSE=[float(w@((q[:, j]-qtrue[:, j])**2)) for j in (0, 1)])
        out['arms'][arm] = dict(partition=partition, row_screen=row, queries=risk_stats, ranking=ranks)
    return out


def diagnose(*, ids, sites, recordings, frames, utility, moving, supported, risks,
             known, easy, reference, benefit, harm):
    arrays = list(map(np.asarray, (ids, sites, recordings, frames, utility, moving, supported,
                                  known, easy, reference, benefit, harm)))
    ids, sites, recordings, frames, utility, moving, supported, known, easy, reference, benefit, harm = arrays
    n = len(ids)
    if (not n or any(a.shape != (n,) for a in arrays) or len(np.unique(ids)) != n or known.dtype != bool
            or not known.any() or any(not np.any(known & (sites == s)) for s in np.unique(sites))):
        raise ValueError('Unique rows with supported sources and aligned arrays required')
    for a in (easy, reference, benefit, harm):
        if not np.array_equal(np.isfinite(a), known) or np.isinf(a).any() or (a[known] < 0).any():
            raise ValueError('Identical unknown label support required')
    if not np.isin(easy[known], [0, 1]).all() or ((benefit[known] > 0) & (harm[known] > 0)).any():
        raise ValueError('Binary easy labels and mutually exclusive benefit/harm required')
    risks = {k: np.asarray(v) for k, v in risks.items()}
    out = _readout(*arrays[:7], risks, *arrays[7:])
    out['by_site'] = {}
    for s in np.unique(sites):
        at = sites == s; a = [x[at] for x in arrays]
        out['by_site'][str(s)] = _readout(*a[:7], {k: v[at] for k, v in risks.items()}, *a[7:])
    out['scope'] = 'fitting_only_independent_sign_screen_not_joint_deployment'
    out['oracle_scope'] = 'realized_same_count_upper_bound_ignores_risk_constraints_not_a_model'
    return out
