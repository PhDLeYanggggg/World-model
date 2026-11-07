"""Support-aware replacement of one member of a fixed-size joint proposal."""
import numpy as np


SHORTLIST = 8
RIDGE = .02


def features(history, bank, floor, crowd, groups):
    v = np.diff(history, axis=1)
    speed = np.linalg.norm(v, axis=-1)
    acceleration = np.linalg.norm(np.diff(v, axis=1), axis=-1)
    disagreement = np.linalg.norm(bank-floor[:, None], axis=-1).max(2)
    dist = np.zeros(len(history))
    contrast = np.zeros(len(history))
    origin = history[:, -1]
    for g in groups:
        g = np.asarray(g)
        if len(g) < 2:
            continue
        delta = origin[g, None]-origin[None, g]
        d = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(d, np.inf)
        j = d.argmin(1)
        dist[g] = d[np.arange(len(g)), j]
        contrast[g] = np.linalg.norm(v[g, -1]-v[g[j], -1], axis=-1)
    return np.log1p(np.column_stack((speed, acceleration, crowd, dist, contrast,
        disagreement.mean(1), disagreement.max(1), disagreement.min(1))))


def train(x, easy, sources, crowd, held):
    keep = sources != held
    x, easy, sources, crowd = x[keep], easy[keep], sources[keep], crowd[keep]
    _, index = np.unique(sources, return_inverse=True)
    w = 1./crowd
    w /= np.bincount(index, weights=w)[index]
    w /= w.sum()
    mean = w @ x
    sd = np.sqrt(w @ (x-mean)**2).clip(.02)
    z = np.column_stack((np.ones(len(x)), (x-mean)/sd))
    reg = np.eye(z.shape[1])*RIDGE
    reg[0, 0] = 0.
    coef = np.linalg.solve(z.T @ (w[:, None]*z)+reg, z.T @ (w*easy))
    return mean, sd, coef


def probability(x, fitted):
    mean, sd, coef = fitted
    return (coef[0] + (x-mean)/sd @ coef[1:]).clip(0., 1.)


def rerank(proposal, utility, scale, legal, support):
    selected = np.flatnonzero(proposal >= 0)
    if len(selected) == 0:
        return proposal.copy(), proposal.copy()
    gain = np.where(legal, utility/scale[:, None], -np.inf)
    current = gain[selected, proposal[selected]]
    replace = selected[current.argmin()]
    anchor = selected[selected != replace]
    best = gain.argmax(1)
    best[replace] = proposal[replace]
    ranked = gain[np.arange(len(gain)), best]
    eligible = np.flatnonzero(np.isfinite(ranked) & ~np.isin(np.arange(len(gain)), anchor))
    top = eligible[np.argsort(-ranked[eligible], kind='stable')[:SHORTLIST]]
    if replace not in top:
        top = np.r_[top[:SHORTLIST-1], replace]
    top = top[np.argsort(-ranked[top], kind='stable')]
    out = []
    for row in (top[0], top[np.argmax(support[top])]):
        use = proposal.copy()
        use[replace] = -1
        use[row] = best[row]
        out.append(use)
    return tuple(out)
