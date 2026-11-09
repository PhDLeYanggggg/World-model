"""Exact two-action relative-motion gains with fixed missing-label charges."""

import numpy as np


def edge_cost(a, b, ref_a, ref_b, y_a, y_b, both):
    rel = a-b
    destination = np.where(both[..., None], y_a-y_b, ref_a-ref_b)
    return np.linalg.norm(rel-destination, axis=-1)


def relative_lower(cv, control, target, valid, rows, paths, first, second, chunk=4096):
    """Use additive incident-edge changes and the exact shared-edge correction."""
    if len(cv) < 2 or np.any(rows[first] == rows[second]):
        raise ValueError('Expected two different actors in every proposal')
    a, b = np.triu_indices(len(cv), 1)
    both = valid[a] & valid[b]
    reference = np.where(both, edge_cost(control[a], control[b], control[a], control[b],
                                        target[a], target[b], both), 0.).sum()
    cv_cost = edge_cost(cv[a], cv[b], control[a], control[b], target[a], target[b], both).sum()
    unary = np.empty(len(rows))
    for lo in range(0, len(rows), 128):
        sl = slice(lo, lo+128)
        r = rows[sl]
        have = valid[r, None] & valid[None]
        args = (control[r, None], control[None], target[r, None], target[None], have)
        delta = edge_cost(paths[sl, None], cv[None], *args)-edge_cost(cv[r, None], cv[None], *args)
        delta[np.arange(len(r)), r] = 0.
        unary[sl] = delta.sum(axis=(1, 2))
    out = np.empty(len(first))
    for lo in range(0, len(first), chunk):
        sl = slice(lo, lo+chunk)
        i, j = first[sl], second[sl]
        a, b = rows[i], rows[j]
        args = (control[a], control[b], target[a], target[b], valid[a] & valid[b])
        shared = (edge_cost(paths[i], paths[j], *args)-edge_cost(paths[i], cv[b], *args)
                  -edge_cost(cv[a], paths[j], *args)+edge_cost(cv[a], cv[b], *args)).sum(1)
        out[sl] = (reference-cv_cost-unary[i]-unary[j]-shared)/both.size
    return out
