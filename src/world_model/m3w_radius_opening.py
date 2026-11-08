"""Past-only, radius-matched first openings and source-weighted regression."""

import numpy as np


def clip_offer(floor, bank, radius=None):
    floor, bank = np.asarray(floor, float), np.asarray(bank, float)
    delta = bank-floor[:, None]
    size = np.linalg.norm(delta, axis=-1).max(2)
    if radius is None and np.any(size > 0.):
        radius = .25*float(np.median(size[size > 0.]))
    if radius is None:
        return bank, np.zeros(size.shape, bool), None
    legal = size >= radius
    ratio = np.divide(radius, size, out=np.zeros_like(size), where=legal)
    return floor[:, None]+ratio[..., None, None]*delta, legal, radius


def offers(floor, bank, groups, incumbent, limit=2):
    """A stop signal at the current query preempts an opening."""
    radius = None
    out = []
    for q in groups:
        if (incumbent[q] >= 0).any():
            break
        paths, legal, radius = clip_offer(floor[q], bank[q], radius)
        i, k = np.nonzero(legal)
        if len(i):
            out.append((q[i], k, paths[i, k], radius))
        if len(out) == limit:
            break
    return out


def ridge(sources, width):
    """Equal source weight, equal record weight within source."""
    g, h = [], []
    for rows in sources:
        if rows:
            g.append(np.mean([x.T@x/len(x) for x, _ in rows], axis=0))
            h.append(np.mean([x.T@y/len(x) for x, y in rows], axis=0))
    if not g:
        return np.zeros(width)
    reg = np.eye(width)*.05
    reg[0, 0] = 0.
    return np.linalg.solve(np.mean(g, axis=0)+reg, np.mean(h, axis=0))


def rank(offer, coefficient):
    return int((offer["x"]@coefficient).argmax())


def query_features(offer, index):
    x = offer["x"]
    return np.r_[x[index], x.mean(0)[1:], np.log1p(len(x)), np.log1p(offer["radius"])]


def fit(by_source, width):
    def table(src):
        return [(np.concatenate([a["x"] for a in r]), np.concatenate([a["quality"] for a in r]))
                for r in by_source[src] if r]

    tables = {s: table(s) for s in by_source}
    pick = ridge(tables.values(), width)
    timing = []
    for s, records in by_source.items():
        inner = ridge([v for k, v in tables.items() if s != k], width)
        parts = []
        for r in records:
            if not r:
                continue
            i = rank(r[0], inner)
            now = r[0]["quality"][i]
            later = r[1]["quality"][rank(r[1], inner)] if len(r) == 2 else 0.
            parts.append((query_features(r[0], i)[None], np.array([later-now])))
        timing.append(parts)
    return pick, ridge(timing, 2*width+1)


def choose(r, pick, wait):
    if not r:
        return None, None, False
    first = rank(r[0], pick)
    delay = bool(query_features(r[0], first)@wait > 0.)
    if delay:
        return (0, first), ((1, rank(r[1], pick)) if len(r) > 1 else None), True
    return (0, first), (0, first), False
