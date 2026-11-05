"""Source-excluded, identity-distinct causal retrieval for risk information."""
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from src.evaluation import m3w_candidate_neighbor_probe as base

K = 32
DISTANCE_EDGES = np.array([0., .25, .5, 1., 2., 4., np.inf])


def fit_bank(x, target, weights, sites, agents, ids, held_source):
    x, target, weights, sites, agents, ids = map(np.asarray, (x, target, weights, sites, agents, ids))
    n = len(x); train = sites != held_source
    if (x.ndim != 2 or target.shape != (n, 2) or any(a.shape != (n,) for a in (weights, sites, agents, ids))
            or not train.any() or train.all() or len(set(ids.tolist())) != n
            or not all(np.isfinite(a).all() for a in (x, target, weights)) or (weights < 0).any()
            or weights[train].sum() <= 0):
        raise ValueError('Aligned finite causal features, separate targets and source-excluded fitting required')
    z, pr = base.normalized(x, train, weights)
    # The same local ID in linked recordings is conservatively merged, not
    # asserted to be an independently verified physical person.
    keys = np.array([str(s)+'|'+str(int(a)) for s,a in zip(sites, agents)])
    donors = np.flatnonzero(train)
    order = np.lexsort((ids[donors], keys[donors])); donors = donors[order]
    names, starts = np.unique(keys[donors], return_index=True)
    stops = np.r_[starts[1:], len(donors)]
    w = weights[train]/weights[train].sum()
    scale = np.sqrt(np.einsum('n,nc->c', w, target[train]**2)).clip(1e-6)
    return dict(**pr, z=z[donors, 1:], donor_positions=donors, donor_ids=ids[donors],
        donor_sources=sites[donors], target=target[donors], weights=weights[donors],
        identities=names, starts=starts, stops=stops, held_source=str(held_source),
        target_scale=scale, prior=w@target[train], k=K)


def neighbors(bank, query_x, query_sources):
    q, sites = np.asarray(query_x), np.asarray(query_sources)
    if (q.ndim != 2 or q.shape[1:] != bank['mean'].shape or sites.shape != (len(q),)
            or not np.isfinite(q).all() or not (sites == bank['held_source']).all()
            or (bank['donor_sources'] == bank['held_source']).any()):
        raise ValueError('Only causal queries from the fully excluded source are allowed')
    if len(bank['identities']) < K:
        raise ValueError('Fewer than32 distinct donor identity buckets: not estimable, not safe')
    z = np.clip((q-bank['mean'])/bank['std'], -8, 8)
    distance = cdist(z, bank['z'], metric='sqeuclidean')
    minima = np.minimum.reduceat(distance, bank['starts'], axis=1)
    groups = np.argsort(minima, axis=1, kind='stable')[:, :K]
    chosen = np.empty_like(groups); selected_distance = np.empty_like(groups, dtype=float)
    for row in range(len(q)):
        for j,g in enumerate(groups[row]):
            lo,hi = bank['starts'][g],bank['stops'][g]
            at = lo+int(np.argmin(distance[row,lo:hi]))
            chosen[row,j] = at
            selected_distance[row,j] = np.sqrt(distance[row,at]/q.shape[1])
    return from_indices(bank, chosen, selected_distance)


def from_indices(bank, chosen, distances):
    chosen, distances = np.asarray(chosen), np.asarray(distances)
    if (chosen.ndim != 2 or chosen.shape[1] != K or chosen.shape != distances.shape
            or not np.issubdtype(chosen.dtype,np.integer) or (chosen < 0).any()
            or (chosen >= len(bank['donor_ids'])).any() or not np.isfinite(distances).all()
            or (distances < 0).any()):
        raise ValueError('Valid saved references and causal distances required')
    groups = np.searchsorted(bank['starts'], chosen, side='right')-1
    if any(len(set(row.tolist())) != K for row in groups):
        raise ValueError('Duplicate donor identity bucket')
    w = bank['weights'][chosen]
    w = w/w.sum(1, keepdims=True)
    y = bank['target'][chosen]
    prediction = np.einsum('nk,nkc->nc', w, y)
    unsafe_probability = np.sum(w*(y > 0).any(-1), axis=1)
    variance = np.einsum('nk,nkc->nc', w, (y-prediction[:,None])**2)
    effective = 1/np.sum(w*w,axis=1)
    return dict(prediction=prediction, local_indices=chosen, donor_ids=bank['donor_ids'][chosen],
        donor_positions=bank['donor_positions'][chosen], identity_indices=groups,
        distances=distances, unsafe_probability=unsafe_probability,
        variance=variance, effective_weight_count=effective)


def permutation(ids, source):
    salt = int.from_bytes(hashlib.sha256(('moving_track_analogue_v1|'+str(source)).encode()).digest()[:8], 'little')
    order = np.argsort(ids, kind='stable')
    donor = np.random.default_rng(salt).permutation(order)
    result = np.empty(len(order), int); result[order] = donor
    return result


def diagnostics(result, target, weights):
    y = np.asarray(target); w = np.asarray(weights, float); w=w/w.sum()
    if y.shape != result['prediction'].shape or not np.isfinite(y).all():
        raise ValueError('Known labels are allowed only after causal retrieval')
    unsafe = (y > 0).any(1)
    p = result['unsafe_probability']
    d = result['distances'][:, 0]
    bins = np.searchsorted(DISTANCE_EDGES[1:-1], d, side='right')
    rows=[]
    for k in range(len(DISTANCE_EDGES)-1):
        at=bins == k; mass=float(w[at].sum())
        rows.append(dict(lower=float(DISTANCE_EDGES[k]),
            upper=None if not np.isfinite(DISTANCE_EDGES[k+1]) else float(DISTANCE_EDGES[k+1]),
            rows=int(at.sum()), weight=mass,
            unsafe_fraction=float(w[at]@unsafe[at]/mass) if mass else None,
            predicted_unsafe_fraction=float(w[at]@p[at]/mass) if mass else None,
            risk_MSE=float(np.sum(w[at,None]*(result['prediction'][at]-y[at])**2)/(2*mass)) if mass else None))
    return dict(distance_slices=rows, nearest_distance_quantiles=np.quantile(d,[0,.25,.5,.75,1]).tolist(),
        neighbor_distance_quantiles=np.quantile(result['distances'],[0,.25,.5,.75,1]).tolist(),
        mixed_neighbor_label_weight=float(w@((p > 0)&(p < 1))),
        cost_probability_Brier=float(w@((p-unsafe)**2)),
        effective_weight_count_quantiles=np.quantile(result['effective_weight_count'],[0,.25,.5,.75,1]).tolist(),
        independent_samples_claimed=False, irreducible_uncertainty_proved=False)


def context_summary(folds):
    arms=('shared64','prior','track32','misaligned32')
    keys=('normalized_MSE','total_joint_cost','false_safe_cost','false_reject_cost')
    means={arm:{k:float(np.mean([f['metrics'][arm][k] for f in folds])) for k in keys} for arm in arms}
    old,new=means['shared64'],means['track32']
    gain=1-new['total_joint_cost']/old['total_joint_cost'] if old['total_joint_cost']>0 else None
    better=new['total_joint_cost'] < means['prior']['total_joint_cost'] and new['total_joint_cost'] < means['misaligned32']['total_joint_cost']
    return dict(source_macro=means,primary_cost_reduction=gain,
        passes_information_screen=bool(gain is not None and gain >= .01 and new['false_safe_cost']<=old['false_safe_cost'] and better),
        safety_certificate=False)
