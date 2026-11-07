"""Causal squared relative-position costs for a fixed binary forecast bank."""
import numpy as np


def geometry(history, floor, candidate, ids):
    """Inputs are absolute observed positions and current-origin-relative paths."""
    origin = history[:, -1]
    dist = np.linalg.norm(origin[:, None]-origin[None], axis=-1)
    edges = set()
    for i in range(len(ids)):
        order = np.lexsort((ids, dist[i]))
        edges.update(tuple(sorted((i, int(j)))) for j in order[order != i][:4])
    i, j = np.array(sorted(edges), dtype=int).reshape(-1, 2).T
    increments = np.diff(history, axis=1)
    scale = 1.+np.linalg.norm(increments, axis=-1).sum(1)
    scale = np.sqrt(scale[i]*scale[j])[:, None, None]
    relative_floor = (floor[i]-floor[j]+(origin[i]-origin[j])[:, None])/scale
    di, dj = (candidate[i]-floor[i])/scale, (candidate[j]-floor[j])/scale
    static = np.concatenate(((origin[i]-origin[j])[:, None], increments[i]-increments[j]), axis=1)/scale
    x = np.concatenate((np.broadcast_to(static[:, None], (len(i), 12, 8, 2)), (di-dj)[:, :, None]), axis=2)
    return i, j, scale, relative_floor, di, dj, x


def observed_residual(g, target, valid):
    i, j, scale, floor, *_ = g
    mask = valid[i] & valid[j]
    residual = np.where(mask[..., None], floor-(target[i]-target[j])/scale, 0.)
    return residual, mask


def moments(g, target, valid):
    """Separate fit-only inputs/labels; equal requested query-edge mass."""
    residual, mask = observed_residual(g, target, valid)
    x = g[-1]
    weight = mask/max(len(x)*12, 1)
    return (np.einsum('etfc,etgc,et->tfg', x, x, weight),
            np.einsum('etfc,etc,et->tf', x, residual, weight))


def costs(g, coefficient, n):
    i, j, _, _, di, dj, x = g
    e = np.einsum('etfc,tf->etc', x, coefficient)
    u = np.zeros(n)
    np.add.at(u, i, (2*e*di+di**2).sum(-1).mean(-1))
    np.add.at(u, j, (-2*e*dj+dj**2).sum(-1).mean(-1))
    interaction = -2*(di*dj).sum(-1).mean(-1)
    return u, interaction


def choose(unary, i, j, interaction, ids):
    """Exact same-count binary decision: two switches with at least three rows."""
    n = len(ids)
    k = min(2, n-1)
    order = np.argsort(ids, kind='stable')
    additive = np.zeros(n, bool); additive[order[np.argsort(unary[order], kind='stable')[:k]]] = True
    coupled = additive.copy()
    if k == 2:
        c = unary[:, None]+unary[None]
        c[i, j] += interaction; c[j, i] += interaction
        c = c[np.ix_(order, order)]
        c[np.tril_indices(n)] = np.inf
        a, b = np.unravel_index(np.argmin(c), c.shape)
        coupled[:] = False; coupled[order[[a, b]]] = True
    return additive, coupled


def error(g, target, valid, selected):
    e, mask = observed_residual(g, target, valid)
    i, j, _, _, di, dj, _ = g
    change = selected[i, None, None]*di-selected[j, None, None]*dj
    value = e+change
    # Compare fixed choices on the same common future support.
    return (float((value[mask]**2).sum()), float(np.linalg.norm(value[mask], axis=-1).sum()), int(mask.sum()))
