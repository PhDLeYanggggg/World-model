"""Source-separated neural fitting roles and past-only, severity-matched vetoes."""
import numpy as np

from src.data_unification.m3w_moving_forecast_data import array_hash
from src.world_model.m3w_multiaction_cost import select_actions

ARMS = ('proposal_self', 'same_data_veto', 'disjoint_data_veto',
        'same_data_matched_self', 'disjoint_data_matched_self')


def partition_jobs(sites, outer, mapping):
    sites = np.asarray(sites)
    names = set(outer['critic_sites'])
    if (set(mapping) != names or set(mapping.values()) != {0, 1, 2}
            or names & set(outer['producer_sites']) or names & set(outer['evaluation_sites'])
            or not names <= set(sites)):
        raise ValueError('Reuse the complete original inner-C mapping; never P or outer H')
    result = []
    for part in range(3):
        selected = sorted(s for s in names if mapping[s] == part)
        ids = np.flatnonzero(np.isin(sites, selected))
        if not selected or not len(ids):
            raise ValueError('Each original inner part must have actual source rows')
        local = dict(outer, evaluation_sites=selected, critic_sites=sorted(names-set(selected)),
            evaluation_scope=dict(rows=len(ids), ids_sha256=array_hash(ids)))
        result.append(dict(part=part, names=selected, routing_job=local))
    return result


def matched_self(own, utility, veto, groups, ids):
    """Match each action/amplitude count within every complete causal query."""
    matched = np.full(len(own), -1, np.int64)
    for group in groups:
        for action in range(1, 7):
            count = int((veto[group] == action).sum())
            candidates = group[own[group] == action]
            order = np.lexsort((ids[candidates], -utility[candidates]))
            take = candidates[order[:count]]
            if len(take) != count:
                raise ValueError('Veto cannot invent an original action')
            matched[take] = action
    return matched


def choices(proposer, same_data, disjoint_data, allowed, groups, ids):
    p, s, d = map(np.asarray, (proposer, same_data, disjoint_data))
    mask, ids = np.asarray(allowed), np.asarray(ids)
    n = len(p)
    if (p.shape != (n, 7, 5) or s.shape != p.shape or d.shape != p.shape
            or mask.shape != (n, 7) or mask.dtype != bool or mask[:, 0].any()
            or ids.shape != (n,) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != n
            or not all(np.isfinite(v).all() and (v >= 0).all() for v in (p, s, d))):
        raise ValueError('Exact aligned causal moments, legal amplitudes and unique IDs required')
    if not groups or any(np.asarray(g).dtype.kind not in 'iu' or not len(g) for g in groups):
        raise ValueError('Complete nonempty integer query groups required')
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n)):
        raise ValueError('Each row must belong to exactly one query')
    own, utility = select_actions(p, mask, arm='physical_neural')
    result = dict(proposal_self=own)
    selected = np.flatnonzero(own >= 0)
    for name, critic in (('same_data', s), ('disjoint_data', d)):
        cost = critic[selected, own[selected]]
        keep = ((cost[:, 2] > 0) & (cost[:, 3] > 0)
            & (cost[:, 1] <= .02*cost[:, 2]) & (cost[:, 4] <= .02*cost[:, 3]))
        veto = np.full(n, -1, np.int64)
        veto[selected[keep]] = own[selected[keep]]
        result[name+'_veto'] = veto
        result[name+'_matched_self'] = matched_self(own, utility, veto, groups, ids)
        for group in groups:
            np.testing.assert_array_equal(np.bincount(veto[group]+1, minlength=8),
                np.bincount(result[name+'_matched_self'][group]+1, minlength=8))
    return {arm: result[arm] for arm in ARMS}
