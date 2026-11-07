"""Pair-grid completion bound using one unknown target per recording/pair/step."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def audit_targets(recordings, agents, frames, target, valid):
    """Check released future labels agree across overlapping occurrences."""
    if (target.shape != (*valid.shape, 2) or valid.dtype != bool
            or any(np.asarray(k).shape != (len(valid),) for k in (recordings, agents, frames))
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Aligned native label roster required')
    t = (np.asarray(frames)[:, None]+np.arange(1, valid.shape[1]+1)).ravel()
    r = np.repeat(recordings, valid.shape[1])
    a = np.repeat(agents, valid.shape[1])
    order = np.lexsort((t, a, r))
    x, y = order[:-1], order[1:]
    same = (t[x] == t[y]) & (a[x] == a[y]) & (r[x] == r[y])
    x, y = x[same], y[same]
    v = valid.ravel()
    if not np.array_equal(v[x], v[y]):
        raise ValueError('Shared target mask conflicts across query origins')
    known = v[x]
    z = target.reshape(-1, 2)
    if not np.array_equal(z[x[known]], z[y[known]]):
        raise ValueError('Shared target position conflicts across query origins')
    return int(same.sum())


def coupling_bound(a, b, weights):
    """Any equal-marginal transport is a certificate; optimal only if equal weights."""
    d = np.linalg.norm(a[:, None]-b[None], axis=2)
    diagonal = np.diag(d)
    original = float(weights@diagonal)
    if np.all(weights == weights[0]):
        i, j = linear_sum_assignment(d)
        return min(original, float(weights[0]*d[i, j].sum()))
    # Symmetric exchanges preserve each row and column mass even when the
    # original query denominators differ. Unexchanged mass stays on diagonal.
    gain = diagonal[:, None]+diagonal[None]-d-d.T
    i, j = np.triu_indices(len(weights), k=1)
    order = np.argsort(-gain[i, j], kind='stable')
    residual = weights.copy()
    cross = 0.
    for k in order:
        x, y = i[k], j[k]
        if gain[x, y] <= 0.:
            break
        v = min(residual[x], residual[y])
        residual[x] -= v
        residual[y] -= v
        cross += v*(d[x, y]+d[y, x])
    return min(original, float(residual@diagonal+cross))


def shared_bound(keys, a, b, weights):
    """Group inside a single recording; keys are ordered pair IDs and absolute step."""
    if (keys.shape != (len(a), 3) or a.shape != b.shape or a.shape != (len(weights), 2)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0.) or np.any(keys[:, 0] >= keys[:, 1])):
        raise ValueError('Canonical pair identities, finite vectors and positive original query weights required')
    if not len(a):
        return 0., 0., 0, 0
    cost = weights*np.linalg.norm(a-b, axis=1)
    prior = float(cost.sum())
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    z = keys[order]
    bounds = np.r_[0, np.flatnonzero(np.any(z[1:] != z[:-1], axis=1))+1, len(z)]
    repeated = np.flatnonzero(np.diff(bounds) > 1)
    savings, occurrences, improved = 0., 0, 0
    for k in repeated:
        take = order[bounds[k]:bounds[k+1]]
        occurrences += len(take)
        reduction = float(cost[take].sum())-coupling_bound(a[take], b[take], weights[take])
        if reduction > 0.:
            savings += reduction
        improved += int(reduction > 1e-12)
    return prior, max(0., prior-savings), occurrences, improved
