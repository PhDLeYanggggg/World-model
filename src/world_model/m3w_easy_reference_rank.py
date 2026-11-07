"""Agent-level easy-reference features from the observed trajectory prefix."""
import numpy as np


def motion(history, groups):
    v = np.diff(history, axis=1)
    speed = np.linalg.norm(v, axis=-1)
    accel = np.linalg.norm(np.diff(v, axis=1), axis=-1)
    three = v[:, -3:].mean(1)
    local = np.zeros((len(v), 3))
    for q in groups:
        drift = np.median(three[q], axis=0)
        distance = np.linalg.norm(history[q, -1, None]-history[q, -1][None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        local[q, 0] = np.linalg.norm(three[q]-drift, axis=-1)
        local[q, 1] = np.linalg.norm(drift)
        local[q, 2] = np.where(len(q)>1, distance.min(1), 0.)
    return np.column_stack((np.log1p(speed), np.log1p(accel), np.log1p(local),
        np.log1p(np.linalg.norm(v[:, -1]-v[:, :3].mean(1), axis=-1)),
        np.log1p(np.linalg.norm(history[:, -1]-history[:, 0], axis=-1))))


def retrieve(groups, legal, score, easy_excess, gain):
    """Prefer separate agents and hold the exact eight-action budget."""
    out = []
    for q in groups:
        i, f = np.where(legal[q])
        i = q[i]
        order = np.lexsort((f, i, -gain[i, f], easy_excess[i, f], -score[i]))
        first, extra, seen = [], [], set()
        for j in order:
            (extra if i[j] in seen else first).append(j)
            seen.add(i[j])
        pick = np.asarray((first+extra)[:8], dtype=int)
        out.append(np.column_stack((i[pick], f[pick])))
    return out


def ridge(x, y, w):
    x = np.column_stack((np.ones(len(x)), x))
    w = w/w.sum()
    pen = np.eye(x.shape[1])*.002
    pen[0, 0] = 0.
    return np.linalg.solve(x.T @ (x*w[:, None])+pen, x.T @ (w*y))
