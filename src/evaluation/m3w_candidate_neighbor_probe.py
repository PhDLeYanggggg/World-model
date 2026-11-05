"""Fixed own-B leave-source-out probes; not an independent model evaluation."""
import numpy as np
from src.evaluation.m3w_decision_semantics import diagnose


def normalized(x, train, weights):
    w = weights[train]/weights[train].sum()
    mean = np.einsum('n,nd->d', w, x[train])
    var = np.einsum('n,nd->d', w, (x[train]-mean)**2)
    std = np.sqrt(var).clip(1e-6)
    z = np.clip((x-mean)/std, -8, 8)
    return np.column_stack((np.ones(len(z)), z)), dict(mean=mean, std=std)


def fit_predict(x, target, train, held, weights, penalty=.01):
    if not train.any() or not held.any() or np.any(train & held):
        raise ValueError('Nonempty source-disjoint train and held subsets required')
    z, pr = normalized(x, train, weights)
    w = weights[train]/weights[train].sum()
    scale = np.sqrt(np.einsum('n,nc->c', w, target[train]**2)).clip(1e-6)
    y = target[train]/scale
    regularizer = np.eye(z.shape[1])*penalty; regularizer[0, 0] = 0
    coefficient = np.linalg.solve(z[train].T@(w[:, None]*z[train])+regularizer, z[train].T@(w[:, None]*y))
    return z[held]@coefficient*scale, dict(**pr, target_scale=scale, coefficient=coefficient)


def balanced_weights(sites, query):
    w = np.zeros(len(sites), float)
    for site in sorted(set(sites)):
        at = sites == site; ids, counts = np.unique(query[at], return_counts=True)
        for q, count in zip(ids, counts):
            w[at & (query == q)] = 1/(len(set(sites))*len(ids)*count)
    return w


def metrics(pred, target, weights, scale):
    w = weights/weights.sum()
    # The existing joint-cost helper retains its original sum-over-channels semantics.
    a = diagnose(pred[:, None], pred[:, None], target[:, None], np.ones((len(pred), 1), bool),
        np.ones(len(pred), bool), w)
    return dict(normalized_MSE=float(np.sum(w[:, None]*((pred-target)/scale)**2)/2), **a)


def probe(common, added, target, sites, query):
    if (common.ndim != 2 or added.shape != (len(common), 24) or target.shape != (len(common), 2)
            or sites.shape != (len(common),) or query.shape != sites.shape
            or not all(np.isfinite(v).all() for v in (common, added, target))):
        raise ValueError('Known fitting outcomes and aligned causal probe features required')
    if len(set(sites)) < 2:
        return dict(status='not_run', reason='Fewer than two own-fitting source groups'), {}
    weights = balanced_weights(sites, query); rows, arrays = [], {}
    for site in sorted(set(sites)):
        held, train = sites == site, sites != site
        row = dict(held_source=str(site), train_sources=sorted(set(sites[train])), rows=int(held.sum()),
            queries=int(len(set(query[held]))), arms={})
        for arm, x in (('history_relations', common), ('plus_candidate_neighbors', np.column_stack((common, added)))):
            pred, state = fit_predict(x, target, train, held, weights)
            row['arms'][arm] = metrics(pred, target[held], weights[held], state['target_scale'])
            arrays[str(site)+'_'+arm+'_predictions'] = pred
            for key, value in state.items():
                arrays[str(site)+'_'+arm+'_'+key] = value
        rows.append(row)
    summary = {arm: {k: float(np.mean([r['arms'][arm][k] for r in rows]))
        for k in ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost')}
        for arm in ('history_relations', 'plus_candidate_neighbors')}
    a, b = summary.values()
    improvement = 1-b['normalized_MSE']/a['normalized_MSE'] if a['normalized_MSE'] > 0 else None
    return dict(status='fresh_run_own_B_leave_source_probe', rows=rows, summary=summary,
        relative_MSE_improvement=improvement, improves_MSE_and_nonworse_cost=bool(improvement is not None
            and improvement >= .01 and b['total_joint_cost'] <= a['total_joint_cost']),
        producer_already_exposed_development=True, independent_confirmation=False), arrays
