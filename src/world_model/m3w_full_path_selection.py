"""Past-only routing over raw or equal-peak action banks."""
from itertools import combinations

import numpy as np

BUDGET = .02
PAIR_WEIGHT = .10
FIXED = ('joint', 'independent', 'uniform', 'no_pair', 'gain_only')


def banks(cv, raw, q):
    offsets = raw-cv[:, None]
    peak = np.linalg.norm(offsets, axis=-1).max(-1)
    allowed = peak[:, :2].max(1) > 0
    raw = np.where(allowed[:, None, None, None], raw, cv[:, None])
    equal = np.broadcast_to(cv[:, None], raw.shape).copy()
    for j in np.unique(q):
        i = np.flatnonzero((q == j) & allowed)
        if len(i):
            # Keep the common scale independent of the neural menu.
            r = np.median(peak[i, :2].max(1))
            legal = peak[i] >= r
            f = np.divide(r, peak[i], out=np.zeros_like(peak[i]), where=legal)
            equal[i] += offsets[i]*f[..., None, None]
    return raw, equal


def apply(cv, bank, choice):
    ix = np.flatnonzero(choice >= 0)
    y = cv.copy()
    y[ix] = bank[ix, choice[ix]]
    np.testing.assert_array_equal((y != cv).any((1, 2)), choice >= 0)
    return y


def choices(h, q, cv, bank, cost, legal, *, fixed):
    """Enumerate at most two distinct recipients; no target or mask input."""
    if not np.isfinite(cost).all():
        raise ValueError('Expected finite value estimates')
    names = FIXED if fixed else ('joint',)
    result = {k: np.full(len(h), -1, dtype=int) for k in names}
    for j in np.unique(q):
        ix = np.flatnonzero(q == j)
        i, k = np.nonzero(legal[ix])
        cap = min(2, len(ix)-1, int(np.bincount(k).max(initial=0)) if fixed else len(set(i)))
        if cap <= 0:
            continue
        row = ix[i]
        val = cost[row, k, :2]
        risk = cost[row, k, 2:4]-BUDGET*cost[row, k, 4:6]
        g, excess = val.min(1), np.maximum(risk, 0).sum(1)
        raw = g-excess
        a, b = np.triu_indices(len(i), 1)
        pair = (i[a] != i[b]) & (cap == 2)
        a, b = a[pair], b[pair]
        local = max(float(np.median(np.linalg.norm(np.diff(h[ix], axis=1), axis=-1)))*12., 1e-8)
        dist = np.linalg.norm(h[row[a], -1]-h[row[b], -1], axis=-1)
        delta = bank[row, k]-cv[row]
        penalty = PAIR_WEIGHT*np.exp(-dist/local)*np.linalg.norm(delta[a]-delta[b], axis=-1).mean(1)
        pval = (val[a]+val[b]).min(1)
        prisk = risk[a]+risk[b]

        def pick(u, p, allow_empty=False):
            u = np.asarray(u)
            p = np.asarray(p)
            best = float(u.max(initial=-np.inf))
            high = float(p.max(initial=-np.inf))
            if allow_empty and max(best, high) <= 0.:
                return []
            if high > best:
                win = int(p.argmax())
                return [a[win], b[win]]
            return [int(u.argmax())]

        if fixed:
            sing = raw if cap == 1 else np.full_like(raw, -np.inf)
            joint = pval-np.maximum(prisk, 0).sum(1)
            plans = dict(
                joint=pick(sing, joint-penalty), no_pair=pick(sing, joint),
                independent=pick(sing, raw[a]+raw[b]),
                uniform=pick(sing, np.where(k[a] == k[b], raw[a]+raw[b], -np.inf)),
                gain_only=pick(g if cap == 1 else np.full_like(g, -np.inf), g[a]+g[b]))
        else:
            plans = dict(joint=pick(np.where((risk <= 0).all(1), g, -np.inf),
                                    np.where((prisk <= 0).all(1), pval-penalty, -np.inf), allow_empty=True))
        for name, actions in plans.items():
            result[name][row[actions]] = k[actions]
    return {k: apply(cv, bank, a) for k, a in result.items()}


def matched(cv, q, paths):
    g = np.unique(q, return_inverse=True)[1]
    widths = []
    for p in paths.values():
        r = np.linalg.norm(p-cv, axis=-1).max(1)
        widths.append((np.bincount(g, weights=r>0), np.bincount(g, weights=r)))
    for n, r in widths[1:]:
        np.testing.assert_array_equal(n, widths[0][0])
        np.testing.assert_allclose(r, widths[0][1], rtol=1e-8, atol=1e-8)
