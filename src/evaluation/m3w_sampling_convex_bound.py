"""Stepwise convex-hull relaxation for velocity schedules with a fixed anchor."""

from itertools import combinations

import numpy as np


def hull_distance(points, vertices):
    """Distance from N-by-T planar points to each row's small convex hull."""
    q, v = np.asarray(points, float), np.asarray(vertices, float)
    if (q.ndim != 3 or q.shape[-1] != 2 or v.ndim != 3 or v.shape[0] != len(q)
            or v.shape[-1] != 2 or v.shape[1] < 2 or not np.isfinite(q).all() or not np.isfinite(v).all()):
        raise ValueError("Aligned finite planar points and at least two vertices required")
    distance = np.full(q.shape[:2], np.inf)
    for i, j in combinations(range(v.shape[1]), 2):
        edge = (v[:, j]-v[:, i])[:, None]
        rel = q-v[:, i, None]
        norm2 = np.square(edge).sum(2)
        t = np.divide((rel*edge).sum(2), norm2, out=np.zeros(q.shape[:2]), where=norm2 > 0.)
        distance = np.minimum(distance, np.linalg.norm(rel-np.clip(t, 0., 1.)[..., None]*edge, axis=2))

    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]

    for i, j, k in combinations(range(v.shape[1]), 3):
        a, b, c = (v[:, m, None] for m in (i, j, k))
        orientation = np.sign(cross(b-a, c-a))
        # Roundoff at a hull edge may only make this lower bound smaller.
        slack = 1e-12*(1.+np.square(np.abs(q)+np.abs(a)+np.abs(b)+np.abs(c)).sum(2))
        inside = ((orientation != 0.) & (cross(b-a, q-a)*orientation >= -slack)
                  & (cross(c-b, q-b)*orientation >= -slack) & (cross(a-c, q-c)*orientation >= -slack))
        distance[inside] = 0.
    return distance
