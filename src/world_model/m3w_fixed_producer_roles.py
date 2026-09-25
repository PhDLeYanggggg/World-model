"""Disjoint development roles without changing the closed confirmation split."""
import numpy as np


def forecaster_baseline(state, roster, fold, seed):
    identity = state['identity']
    if (state['step'] != 4000 or identity['kind'] != 'single'
            or identity['fold'] != fold or identity['seed'] != seed
            or sorted(identity['fit_sites']) != sorted(roster)
            or identity['baseline_index'] not in range(6)):
        raise ValueError('Frozen forecaster identity differs from producer role')
    return identity['baseline_index']


def role_rosters(halves):
    if set(halves) != {'0', '1', '2'}:
        raise ValueError('Three pre-existing source folds required')
    rosters = []
    for i in range(3):
        pair = halves[str(i)]
        if len(pair) != 2 or any(len(x) != 2 for x in pair):
            raise ValueError('Each producer fold has two two-source halves')
        roster = sorted(pair[0]+pair[1])
        if len(set(roster)) != 4: raise ValueError('Repeated producer source')
        rosters.append(roster)
    if len(set(sum(rosters, []))) != 12:
        raise ValueError('Producer folds must partition the twelve opened sources')
    return rosters


def role_indices(sites, rosters, producer, controller):
    sites = np.asarray(sites)
    if (len(rosters) != 3 or any(len(set(r)) != 4 for r in rosters)
            or len(set(sum(rosters, []))) != 12 or set(sites) != set(sum(rosters, []))
            or producer not in range(3) or controller not in range(3) or producer == controller):
        raise ValueError('Disjoint producer/controller/readout roles on opened sources only')
    readout = next(i for i in range(3) if i not in (producer, controller))
    ids = [np.flatnonzero(np.isin(sites, rosters[i])) for i in (producer, controller, readout)]
    if any(len(x) == 0 for x in ids) or len(np.unique(np.concatenate(ids))) != len(sites):
        raise ValueError('Every indexed row must belong to exactly one role')
    return dict(producer=ids[0], controller=ids[1], readout=ids[2], readout_fold=readout)


def bounded_ridge_scores(raw, envelope, task):
    raw, envelope = np.asarray(raw, float), np.asarray(envelope, float)
    if (raw.shape != (len(envelope), 2) or not np.isfinite(raw).all()
            or not np.isfinite(envelope).all() or np.any(envelope < 0) or task not in ('utility', 'risk')):
        raise ValueError('Finite causal score/envelope arrays required')
    scores = np.maximum(raw, 0)
    scores[:, 1] = np.minimum(scores[:, 1], envelope)
    if task == 'utility': scores[:, 0] = np.minimum(scores[:, 0], envelope)
    return scores


def label_free_replay(scores, moving, budget=.02):
    """Independent scalar implementation, not the vectorized deployment code."""
    u, r = scores
    if len(u) != len(r) or len(u) != len(moving): raise ValueError('Misaligned score arrays')
    return np.asarray([bool(m and a[0] > a[1] and b[1] <= budget*b[0])
                       for a, b, m in zip(u, r, moving)], bool)
