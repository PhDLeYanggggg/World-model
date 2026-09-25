"""Factor fixed support boxes without refitting thresholds or consuming outcomes."""
import numpy as np

from src.world_model.m3w_causal_abstention import matched_count, random_priority

GUARDS = ('history', 'disagreement', 'separate', 'joint')
REASONS = ('retained', 'history_only_failure', 'disagreement_only_failure',
           'both_marginals_fail', 'source_overlap_failure', 'outside_stop_policy')


def memberships(x, states, fitted):
    x, states = np.asarray(x), np.asarray(states)
    if x.shape != (len(states), 3) or not np.isfinite(x).all():
        raise ValueError('Three finite causal features required')
    sources = fitted['sources']; h = np.zeros((len(x), len(sources)), bool); d = h.copy()
    seen = set()
    for b in fitted['boxes']:
        key = (b['source'], b['state'])
        if key in seen or b['source'] not in sources or b['count'] < 32:
            raise ValueError('Unique, sufficiently supported source/state boxes required')
        seen.add(key); j = sources.index(b['source'])
        inside = (x >= b['low']) & (x <= b['high']); state = states == b['state']
        h[:, j] |= state & inside[:, :2].all(1)
        d[:, j] |= state & inside[:, 2]
    return h, d


def guards_and_reasons(stop, history, disagreement):
    h = history.sum(1) >= 2; d = disagreement.sum(1) >= 2
    joint = (history & disagreement).sum(1) >= 2
    masks = dict(history=stop & h, disagreement=stop & d,
                 separate=stop & h & d, joint=stop & joint)
    reason = np.full(len(stop), 5, np.int8)
    reason[stop & joint] = 0
    reason[stop & ~h & d] = 1
    reason[stop & h & ~d] = 2
    reason[stop & ~h & ~d] = 3
    reason[stop & h & d & ~joint] = 4
    assert not (joint & ~(h & d)).any()
    return masks, reason


def decisions(stop, x, states, fitted, query, risk, ids, seed):
    stop = np.asarray(stop, bool)
    if (stop & (states != 2)).any(): raise ValueError('Stopping floor must be retained')
    h, d = memberships(x, states, fitted)
    guards, reason = guards_and_reasons(stop, h, d)
    priority = np.divide(risk[:, 1], risk[:, 0], out=np.full(len(ids), np.inf), where=risk[:, 0] > 0)
    out = dict(stop=stop.copy())
    for name, chosen in guards.items():
        out[name] = chosen
        out[name+'_risk'] = matched_count(stop, chosen, query, priority, ids)
        out[name+'_random'] = matched_count(stop, chosen, query, random_priority(ids, seed), ids)
    return out, reason


def independent_verify(choices, reasons, stop, x, states, fitted, query, risk, ids, seed):
    n = len(ids); guards = {g: np.zeros(n, bool) for g in GUARDS}; expected_reasons = np.full(n, 5, np.int8)
    for i in np.flatnonzero(stop):
        hs, ds = set(), set()
        for b in fitted['boxes']:
            if states[i] != b['state']: continue
            if all(b['low'][j] <= x[i, j] <= b['high'][j] for j in (0, 1)): hs.add(b['source'])
            if b['low'][2] <= x[i, 2] <= b['high'][2]: ds.add(b['source'])
        h, d, joint = len(hs) >= 2, len(ds) >= 2, len(hs & ds) >= 2
        guards['history'][i], guards['disagreement'][i] = h, d
        guards['separate'][i], guards['joint'][i] = h and d, joint
        expected_reasons[i] = 0 if joint else 1 if not h and d else 2 if h and not d else 3 if not h and not d else 4
    np.testing.assert_array_equal(reasons, expected_reasons)
    np.testing.assert_array_equal(choices['stop'], stop)
    buckets = {}
    for i in np.flatnonzero(stop): buckets.setdefault(int(query[i]), []).append(int(i))
    ratio = np.divide(risk[:, 1], risk[:, 0], out=np.full(n, np.inf), where=risk[:, 0] > 0)
    priorities = dict(risk=ratio, random=random_priority(ids, seed))
    for name, chosen in guards.items():
        np.testing.assert_array_equal(chosen, choices[name])
        for control, priority in priorities.items():
            expected = np.zeros(n, bool)
            for bucket in buckets.values():
                count = sum(bool(chosen[i]) for i in bucket)
                expected[sorted(bucket, key=lambda i: (priority[i], ids[i]))[:count]] = True
            np.testing.assert_array_equal(expected, choices[name+'_'+control])
    assert all(not (v & ~stop).any() for v in choices.values())
    return 13


def partition_ledger(reason, neural_error, floor_error, sites, mask):
    out = {}
    delta = neural_error-floor_error
    for source in sorted(set(sites)):
        known = mask & (sites == source) & np.isfinite(delta)
        denominator = float(floor_error[known].sum()); rows = {}
        for i, name in enumerate(REASONS):
            use = known & (reason == i)
            harm, benefit = float(np.maximum(delta[use], 0).sum()), float(np.maximum(-delta[use], 0).sum())
            rows[name] = dict(indexed_rows=int((mask & (sites == source) & (reason == i)).sum()),
                known_rows=int(use.sum()), harm=harm, benefit=benefit,
                harm_pp=100*harm/denominator if denominator > 0 else None,
                benefit_pp=100*benefit/denominator if denominator > 0 else None)
        out[source] = dict(floor_error_sum=denominator, categories=rows)
    return out
