"""Budget missing positions by unresolved pair-disagreement mass."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array, csr_array, vstack


def aggregate(links, mass):
    """Store singleton dependencies in column zero with sentinel -1 in column one."""
    p = np.sort(links, axis=1)
    if (p.shape != (len(mass), 2) or not len(mass) or np.any(p[:, 0] < -1)
            or np.any(p[:, 1] < 0) or np.any(p[:, 0] == p[:, 1])
            or not np.isfinite(mass).all() or np.any(mass <= 0.)):
        raise ValueError('Each positive mass needs one or two distinct missing positions')
    keys, inv = np.unique(p, return_inverse=True, axis=0)
    keys = np.where(keys < 0, -1, np.searchsorted(np.unique(keys[keys >= 0]), keys))
    keys = np.where(keys[:, :1] < 0, keys[:, ::-1], keys)
    return keys, np.bincount(inv, weights=mass)


def curves(keys, mass):
    n = int(keys.max())+1
    one = keys[:, 1] < 0
    a, b = keys[~one].T
    cost = mass[~one]
    singleton = np.bincount(keys[one, 0], weights=mass[one], minlength=n)
    incident = singleton+np.bincount(np.r_[a, b], weights=np.r_[cost, cost], minlength=n)
    order, selected = [], np.zeros(n, dtype=bool)
    while len(order) < n:
        single = singleton.copy()
        np.add.at(single, a, cost*selected[b])
        np.add.at(single, b, cost*selected[a])
        single[selected] = -np.inf
        node = int(single.argmax())
        both = (single[a]+single[b]+cost)/2.
        if len(both) and both.max() > single[node]:
            k = int(both.argmax())
            take = sorted((int(a[k]), int(b[k])), key=lambda x: (-single[x], x))
        else:
            take = [node]
        selected[take] = True
        order.extend(take)

    def coverage(sequence):
        rank = np.empty(n, dtype=int)
        rank[sequence] = np.arange(1, n+1)
        end = np.maximum(rank[keys[:, 0]], np.where(one, 0, rank[keys[:, 1]]))
        values = np.cumsum(np.bincount(end, weights=mass, minlength=n+1))
        np.testing.assert_allclose(values[-1], mass.sum(), atol=1e-12)
        return values/mass.sum()

    return {k: coverage(x) for k, x in dict(chronological=np.arange(n),
            incident_mass=np.argsort(-incident, kind='stable'), dependency_greedy=np.asarray(order)).items()}


def minimum_relaxation(keys, mass, fraction):
    """Continuous relaxation of the minimum-position AND-coverage problem."""
    n, m = int(keys.max())+1, len(mass)
    i, j = np.nonzero(keys >= 0)
    rows = np.arange(len(i))
    # z_term <= x_endpoint, for every missing endpoint of every term.
    mat = coo_array((np.r_[-np.ones(len(i)), np.ones(len(i))],
                     (np.r_[rows, rows], np.r_[keys[i, j], n+i])), shape=(len(i), n+m))
    coverage = csr_array(np.r_[np.zeros(n), -mass/mass.sum()][None])
    fit = linprog(np.r_[np.ones(n), np.zeros(m)], A_ub=vstack((mat, coverage), format='csr'),
                  b_ub=np.r_[np.zeros(len(i)), -fraction], bounds=(0., 1.), method='highs')
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError('Coverage budget relaxation failed')
    return float(fit.fun)
