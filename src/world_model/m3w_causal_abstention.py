"""Past-only abstention and same-query count-matched controls, not a risk certificate."""
import hashlib

import numpy as np

FEATURES = ('latest_to_mean_speed', 'mean_turn_radians', 'rollout_disagreement_to_path')


def causal_features(history, candidate, floor):
    h, n, d = (np.asarray(v, dtype=np.float64) for v in (history, candidate, floor))
    if h.ndim != 3 or h.shape[1:] != (8, 2) or n.shape != d.shape or n.shape[1:] != (12, 2):
        raise ValueError('Expected aligned obs8/pred12 coordinate arrays')
    if len(h) != len(n) or not all(np.isfinite(v).all() for v in (h, n, d)):
        raise ValueError('Nonfinite or unaligned causal inputs')
    delta = np.diff(h, axis=1); speed = np.linalg.norm(delta, axis=2)
    mean = speed.mean(1); latest = speed[:, -1]
    pair = (speed[:, :-1] > 0) & (speed[:, 1:] > 0)
    dot = (delta[:, :-1]*delta[:, 1:]).sum(2)
    denom = speed[:, :-1]*speed[:, 1:]
    cosine = np.divide(dot, denom, out=np.ones_like(dot), where=pair)
    angles = np.arccos(np.clip(cosine, -1, 1))*pair
    turn = np.divide(angles.sum(1), pair.sum(1), out=np.zeros(len(h)), where=pair.sum(1) > 0)
    disagreement = np.linalg.norm(n-d, axis=2).mean(1)
    x = np.column_stack((np.divide(latest, mean, out=np.zeros(len(h)), where=mean > 0), turn,
        np.divide(disagreement, 12*mean, out=np.zeros(len(h)), where=mean > 0)))
    # States use observed history, never future stationarity or a zero-error label.
    state = np.where(latest > 0, 2, np.where(mean > 0, 1, 0)).astype(np.int8)
    return x, state


def fit_support(x, states, sources, known, *, minimum_rows=32, quantiles=(.01, .99)):
    x, states, sources, known = map(np.asarray, (x, states, sources, known))
    if x.shape != (len(states), 3) or any(len(v) != len(x) for v in (sources, known)):
        raise ValueError('Unaligned fitting support')
    boxes = []
    for source in sorted(set(sources.tolist())):
        for state in range(3):
            use = (sources == source) & (states == state) & known
            count = int(use.sum())
            if count < minimum_rows:
                continue
            low, high = np.quantile(x[use], quantiles, axis=0)
            boxes.append(dict(source=str(source), state=state, count=count,
                              low=low.tolist(), high=high.tolist()))
    return dict(features=list(FEATURES), sources=sorted(set(sources.tolist())),
                minimum_rows=minimum_rows, quantiles=list(quantiles), boxes=boxes)


def support_count(x, states, fitted):
    count = np.zeros(len(x), np.int16)
    for b in fitted['boxes']:
        count += ((states == b['state']) & (x >= b['low']).all(1) & (x <= b['high']).all(1))
    return count


def query_ids(recordings, frames):
    # A query is one observed frame of one recording, not a whole future sequence.
    keys = np.rec.fromarrays([np.asarray(recordings, dtype=str), np.asarray(frames)], names=('recording', 'frame'))
    return np.unique(keys, return_inverse=True)[1]


def random_priority(ids, seed):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}:{int(i)}'.encode('ascii')).digest()[:8], 'big')
                     for i in ids], dtype=np.uint64)


def matched_count(original, guard, query, priority, ids):
    original, guard = np.asarray(original, bool), np.asarray(guard, bool)
    if (guard & ~original).any():
        raise ValueError('Guard must only remove interventions')
    idx = np.flatnonzero(original)
    result = np.zeros(len(original), bool)
    if not len(idx):
        return result
    order = idx[np.lexsort((ids[idx], priority[idx], query[idx]))]
    q = query[order]
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1])+1]
    ranks = np.arange(len(order))-np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    counts = np.bincount(query[guard], minlength=int(query.max())+1)
    result[order] = ranks < counts[q]
    return result


def variants(original, x, states, fitted, query, risk, ids, seed, *, minimum_sources=2):
    support = support_count(x, states, fitted) >= minimum_sources
    guards = dict(stop=states == 2, support=support, combined=(states == 2) & support)
    risk_priority = np.divide(risk[:, 1], risk[:, 0], out=np.full(len(ids), np.inf), where=risk[:, 0] > 0)
    priorities = dict(risk=risk_priority, random=random_priority(ids, seed))
    out = dict(original=np.asarray(original, bool).copy())
    for name, guard in guards.items():
        chosen = original & guard
        out[name] = chosen
        for control, priority in priorities.items():
            out[name+'_'+control] = matched_count(original, chosen, query, priority, ids)
    return out


def verify_variants(choices, original, x, states, fitted, query, risk, ids, seed):
    """Separately loop over boxes/queries to verify the vectorized decision bank."""
    count = np.array([sum(s == b['state'] and all(l <= v <= u for v, l, u in zip(row, b['low'], b['high']))
                         for b in fitted['boxes']) for row, s in zip(x, states)])
    guards = dict(stop=states == 2, support=count >= 2, combined=(states == 2) & (count >= 2))
    np.testing.assert_array_equal(choices['original'], original)
    r = np.divide(risk[:, 1], risk[:, 0], out=np.full(len(ids), np.inf), where=risk[:, 0] > 0)
    priorities = dict(risk=r, random=random_priority(ids, seed))
    buckets = {}
    for i in np.flatnonzero(original):
        buckets.setdefault(int(query[i]), []).append(int(i))
    for name, guard in guards.items():
        np.testing.assert_array_equal(choices[name], original & guard)
        for control, priority in priorities.items():
            expected = np.zeros(len(ids), bool)
            for bucket in buckets.values():
                k = sum(bool(guard[i]) for i in bucket)
                selected = sorted(bucket, key=lambda i: (priority[i], ids[i]))[:k]
                expected[selected] = True
            np.testing.assert_array_equal(choices[name+'_'+control], expected)
    for v in choices.values():
        if (v & ~original).any():
            raise ValueError('Abstention increased interventions')
    return len(choices)
