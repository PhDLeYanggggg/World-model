"""Frozen-policy error accounting; no deployable oracle or new risk threshold."""
import numpy as np

BUDGET = .02
ARMS = ('local', 'anchor', 'old_risk', 'utility_continue', 'joint_risk', 'isolated_risk')


def actions(prediction, eligible):
    p = np.asarray(prediction, float)
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all():
        raise ValueError('Finite five-moment predictions required')
    return np.asarray(eligible, bool) & (p[:, 0] > p[:, 1]) & (
        p[:, 1] <= BUDGET*p[:, 2]) & (p[:, 4] <= BUDGET*p[:, 3])


def radius(x, preprocessing):
    """Unclipped standardized radius, not a density or OOD certificate."""
    z = (np.asarray(x, float)-preprocessing['mean'])/preprocessing['std']
    if not np.isfinite(z).all():
        raise ValueError('Finite causal features required')
    return np.sqrt(np.mean(z*z, axis=1))


def fitting_bins(values, eligible):
    values = np.asarray(values, float)
    pool = values[np.asarray(eligible, bool)]
    if not len(pool) or not np.isfinite(pool).all():
        raise ValueError('Nonempty finite inner-fit causal pool required')
    return np.quantile(pool, [.5, .9, .99]).tolist()


def bin_ids(values, edges):
    if len(edges) != 3 or np.any(np.diff(edges) < 0):
        raise ValueError('Fixed monotone fitting-only edges required')
    return np.searchsorted(edges, values, side='right')


def harm_upper(anchor, envelope):
    """Supremum as both isolated branch shifts approach +5 (not attainable at finite logits)."""
    p, env = np.asarray(anchor, float), np.asarray(envelope, float)
    cap = np.maximum(env-p[:, 0], 0)
    h = np.minimum(p[:, 1], cap)
    u = np.expm1(5.)
    tiny = np.finfo(np.float32).tiny
    maximum_h = h+h*(cap-h)*u/np.maximum(cap+h*u, tiny)
    q = np.clip(p[:, 4]/np.maximum(p[:, 1], tiny), 0, 1)
    maximum_q = q*(1+u)/(1+q*u)
    return np.column_stack((np.minimum(maximum_h, cap), np.minimum(maximum_h*maximum_q, cap)))


def accounting(prediction, truth, cohort, scale):
    p, y, cohort = np.asarray(prediction, float), np.asarray(truth, float), np.asarray(cohort, bool)
    if p.shape != y.shape or p.shape[1] != 5 or cohort.shape != (len(y),) or scale <= 0:
        raise ValueError('Aligned five-moment rows and positive frozen scale required')
    known = np.isfinite(y).all(1)
    at = cohort & known
    pp, yy = p[at]/scale, y[at]/scale
    result = dict(rows=int(cohort.sum()), known=int(at.sum()), unknown=int((cohort & ~known).sum()))
    for label, h, r in (('all', 1, 2), ('easy', 4, 3)):
        pred = pp[:, h]-BUDGET*pp[:, r]
        harm = yy[:, h]-pp[:, h]
        reference = BUDGET*(pp[:, r]-yy[:, r])
        obs = yy[:, h]-BUDGET*yy[:, r]
        np.testing.assert_allclose(obs, pred+harm+reference, atol=1e-10, rtol=1e-10)
        bad = obs > 1e-12
        # These replacements use outcomes and only explain a frozen cohort; never infer a policy.
        result[label] = dict(predicted_excess=float(pred.sum()), observed_excess=float(obs.sum()),
            harm_underestimate=float(harm.sum()), reference_overestimate=float(reference.sum()),
            positive_harm_underestimate=float(np.maximum(harm, 0).sum()),
            positive_reference_overestimate=float(np.maximum(reference, 0).sum()),
            predicted_harm=float(pp[:, h].sum()), observed_harm=float(yy[:, h].sum()),
            predicted_reference=float(pp[:, r].sum()), observed_reference=float(yy[:, r].sum()),
            violating_rows=int(bad.sum()),
            violating_rows_still_allowed_with_true_reference=int((bad & (pp[:, h] <= BUDGET*yy[:, r])).sum()),
            violating_rows_still_allowed_with_true_harm=int((bad & (yy[:, h] <= BUDGET*pp[:, r])).sum()),
            zero_reference_rows=int((yy[:, r] == 0).sum()),
            harmed_zero_reference_rows=int(((yy[:, r] == 0) & (yy[:, h] > 0)).sum()))
    return result


def ceiling_accounting(anchor, envelope, truth, cohort, scale):
    known = np.isfinite(truth).all(1) & cohort
    upper, y = harm_upper(anchor, envelope)[known]/scale, truth[known]/scale
    out = dict(known=int(known.sum()))
    for name, i, col in (('all', 0, 1), ('easy', 1, 4)):
        outside = y[:, col] > upper[:, i]+1e-7
        out[name] = dict(above_supremum_rows=int(outside.sum()),
            observed_harm=float(y[:, col].sum()),
            observed_harm_above_supremum=float(y[outside, col].sum()),
            minimum_shortfall=float(np.maximum(y[:, col]-upper[:, i], 0).sum()))
    return out


def view(predictions, truth, envelope, eligible, radial, edges, scale):
    if set(predictions) != set(ARMS):
        raise ValueError('All six controls required')
    selected = {a: actions(p, eligible) for a, p in predictions.items()}
    bins = bin_ids(radial, edges)
    out = {}
    for arm, p in predictions.items():
        cohorts = dict(eligible=eligible, own=selected[arm], anchor=selected['anchor'])
        out[arm] = dict(cohorts={n: accounting(p, truth, mask, scale) for n, mask in cohorts.items()},
            own_radius_bins=[accounting(p, truth, selected[arm] & (bins == b), scale) for b in range(4)])
    ceilings = {n: ceiling_accounting(predictions['anchor'], envelope, truth, mask, scale)
        for n, mask in dict(eligible=eligible, anchor=selected['anchor'], isolated=selected['isolated_risk']).items()}
    return dict(arms=out, isolated_architecture_ceiling=ceilings,
        rows=len(truth), unknown_rows=int((~np.isfinite(truth).all(1)).sum()),
        radius_edges=edges, outcome_filter_at_inference=False, counterfactuals_not_deployable=True)


def validate_view(result):
    for arm in ARMS:
        own = result['arms'][arm]['cohorts']['own']
        parts = result['arms'][arm]['own_radius_bins']
        for key in ('rows', 'known', 'unknown'):
            if own[key] != sum(v[key] for v in parts):
                raise ValueError('Radius accounting lost rows')
        for cohort in [*result['arms'][arm]['cohorts'].values(), *parts]:
            if cohort['rows'] != cohort['known']+cohort['unknown']:
                raise ValueError('Unknown outcomes cannot disappear')
            for kind in ('all', 'easy'):
                v = cohort[kind]
                np.testing.assert_allclose(v['observed_excess'], v['predicted_excess']+
                    v['harm_underestimate']+v['reference_overestimate'], atol=1e-7, rtol=1e-9)
        for kind in ('all', 'easy'):
            for key, value in own[kind].items():
                np.testing.assert_allclose(value, sum(v[kind][key] for v in parts), atol=1e-7, rtol=1e-9)


def summarize(rows):
    actual = {(r['scope'], r['context'], r['source'], r['seed']) for r in rows}
    if len(rows) != 144 or len(actual) != 144:
        raise ValueError('All144 view identities required')
    for r in rows:
        validate_view(r['result'])
    result = dict(views=144, new_training=False, independent_roles_read=False,
        diagnostic_only=True, deployment_changed=False, radius_is_support_proxy_only=True)
    for scope in ('fit', 'held'):
        group = [r for r in rows if r['scope'] == scope]
        if len(group) != 72 or len({r['source'] for r in group}) != 12:
            raise ValueError('All72 views and12 sources required')
        summary = {}
        for arm in ARMS:
            aggregate = {}
            for cohort in ('eligible', 'own', 'anchor'):
                cc = [r['result']['arms'][arm]['cohorts'][cohort] for r in group]
                d = {k: sum(v[k] for v in cc) for k in ('rows', 'known', 'unknown')}
                for kind in ('all', 'easy'):
                    d[kind] = {k: sum(v[kind][k] for v in cc) for k in cc[0][kind]}
                    d[kind]['violating_views'] = sum(v[kind]['observed_excess'] > 1e-10 for v in cc)
                aggregate[cohort] = d
            aggregate['radius_known_easy_violating_rows'] = [sum(
                r['result']['arms'][arm]['own_radius_bins'][b]['easy']['violating_rows'] for r in group) for b in range(4)]
            aggregate['radius_known_selected'] = [sum(
                r['result']['arms'][arm]['own_radius_bins'][b]['known'] for r in group) for b in range(4)]
            summary[arm] = aggregate
        result[scope] = summary
        result[scope+'_ceilings'] = {cohort: {kind: {k: sum(
            r['result']['isolated_architecture_ceiling'][cohort][kind][k] for r in group)
            for k in group[0]['result']['isolated_architecture_ceiling'][cohort][kind]}
            for kind in ('all', 'easy')} for cohort in ('eligible', 'anchor', 'isolated')}
    return result
