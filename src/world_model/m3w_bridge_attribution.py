"""Motion-only action pairs and label-free matched-query risk rankings."""
import hashlib
import numpy as np
from src.world_model.m3w_dual_event_bridge import features, choices

POLICIES = ('reference', 'candidate', 'neural', 'ridge', 'neural_utility_ridge_risk',
            'ridge_utility_neural_risk', 'neural_common', 'ridge_common',
            'neural_matched', 'ridge_matched', 'hash_matched')


def motion_pair(geometry, cv, damped, easy_bits, all_bits):
    b, d = np.asarray(cv), np.asarray(damped)
    eb, ab = np.asarray(easy_bits), np.asarray(all_bits)
    if (eb.dtype != bool or ab.dtype != bool or eb.shape != (len(b),) or ab.shape != eb.shape
            or b.shape != (len(b), 12, 2) or d.shape != b.shape):
        raise ValueError('Aligned causal floor bits and twelve-step motion rollouts required')
    r, p = np.where(eb[:, None, None], d, b), np.where(ab[:, None, None], d, b)
    # Remove neural-policy bits as well as neural trajectories from the feature API.
    bits = np.column_stack((np.zeros((len(b), 2), bool), eb, ab))
    x, env = features(geometry, b, r, p, bits)
    return x, env, r, p


def query_groups(sites, recordings, frames):
    s, r, f = map(np.asarray, (sites, recordings, frames))
    if s.ndim != 1 or r.shape != s.shape or f.shape != s.shape:
        raise ValueError('Aligned past query keys required')
    groups = {}
    for i, key in enumerate(zip(s.tolist(), r.tolist(), f.tolist())):
        groups.setdefault(key, []).append(i)
    return [np.asarray(v, np.int64) for v in groups.values()]


def decisions(neural_utility, neural_risk, ridge_utility, ridge_risk,
              moving, envelope, groups, row_ids):
    nu, nr, ru, rr = map(np.asarray, (neural_utility, neural_risk, ridge_utility, ridge_risk))
    n = len(moving); ids = np.asarray(row_ids)
    if ids.shape != (n,) or len(set(ids.tolist())) != n:
        raise ValueError('Unique stable past-index row identifiers required')
    covered = np.concatenate(groups) if groups else np.array([], int)
    if not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError('Queries must partition all past-indexed rows exactly once')
    def gate(u, r): return choices(u, r, r, moving, envelope, arm='all_risk_only')
    out = dict(reference=np.zeros(n, bool), candidate=np.ones(n, bool),
        neural=gate(nu, nr), ridge=gate(ru, rr),
        neural_utility_ridge_risk=gate(nu, rr), ridge_utility_neural_risk=gate(ru, nr))
    common = (np.asarray(moving, bool) & (envelope > 0) & (nu[:, 0] > nu[:, 1])
              & (ru[:, 0] > ru[:, 1]) & (nr[:, 0] > 0) & (rr[:, 0] > 0))
    out['neural_common'] = out['neural'] & common
    out['ridge_common'] = out['ridge'] & common
    for k in ('neural_matched', 'ridge_matched', 'hash_matched'): out[k] = np.zeros(n, bool)
    ratio_n = np.divide(nr[:, 1], nr[:, 0], out=np.full(n, np.inf), where=nr[:, 0] > 0)
    ratio_r = np.divide(rr[:, 1], rr[:, 0], out=np.full(n, np.inf), where=rr[:, 0] > 0)
    hash_order = np.asarray([hashlib.sha256(f'bridge-attribution-v1|{i}'.encode()).hexdigest() for i in ids])
    counts = []
    for query in groups:
        pool = query[common[query]]
        count = min(int(out['neural_common'][query].sum()), int(out['ridge_common'][query].sum()))
        for name, score in (('neural_matched', ratio_n), ('ridge_matched', ratio_r), ('hash_matched', hash_order)):
            selected = pool[np.lexsort((ids[pool], score[pool]))[:count]]
            out[name][selected] = True
        counts.append(count)
    if set(out) != set(POLICIES): raise ValueError('Incomplete registered control family')
    if np.any(out['neural_matched'] & ~out['neural']) or np.any(out['ridge_matched'] & ~out['ridge']):
        raise ValueError('Ranked prefix must preserve its own predicted threshold')
    return out, common, np.asarray(counts, np.int64)


def independent_replay(nu, nr, ru, rr, moving, envelope, groups, row_ids):
    """Scalar ranking verification independent of the vectorized decision code."""
    n = len(moving)
    def allowed(u, r, i):
        return bool(moving[i] and envelope[i] > 0 and u[i, 0] > u[i, 1] and r[i, 1] <= .02*r[i, 0])
    out = dict(reference=np.zeros(n, bool), candidate=np.ones(n, bool))
    for name, u, r in (('neural', nu, nr), ('ridge', ru, rr),
                       ('neural_utility_ridge_risk', nu, rr), ('ridge_utility_neural_risk', ru, nr)):
        out[name] = np.asarray([allowed(u, r, i) for i in range(n)], bool)
    common = [bool(moving[i] and envelope[i] > 0 and nu[i, 0] > nu[i, 1] and ru[i, 0] > ru[i, 1]
                   and nr[i, 0] > 0 and rr[i, 0] > 0) for i in range(n)]
    out['neural_common'] = out['neural'] & common; out['ridge_common'] = out['ridge'] & common
    for name in ('neural_matched', 'ridge_matched', 'hash_matched'): out[name] = np.zeros(n, bool)
    for query in groups:
        pool = [int(i) for i in query if common[i]]
        k = min(sum(out['neural_common'][query]), sum(out['ridge_common'][query]))
        norder = sorted(pool, key=lambda i: (float(nr[i, 1])/float(nr[i, 0]), int(row_ids[i])))
        rorder = sorted(pool, key=lambda i: (float(rr[i, 1])/float(rr[i, 0]), int(row_ids[i])))
        horder = sorted(pool, key=lambda i: (hashlib.sha256(f'bridge-attribution-v1|{row_ids[i]}'.encode()).hexdigest(), int(row_ids[i])))
        for name, order in zip(('neural_matched', 'ridge_matched', 'hash_matched'), (norder, rorder, horder)):
            out[name][order[:k]] = True
    return out
