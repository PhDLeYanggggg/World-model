"""Compare forecast directions using only outcomes inside the observed window."""

import numpy as np


PREFIXES = (4, 5, 6, 7)
PADDING = ("linear", "stationary")


def prefix_input(history, length, mode):
    if history.shape[1:] != (8, 2) or length not in PREFIXES or mode not in PADDING:
        raise ValueError("Expected an eight-step history and a registered prefix")
    prefix = history[:, :length]
    v = prefix[:, 1:2]-prefix[:, :1] if mode == "linear" else np.zeros_like(prefix[:, :1])
    fill = prefix[:, :1]+np.arange(length-8, 0)[None, :, None]*v
    return np.concatenate((fill, prefix), axis=1)


def replay(history, forecast, match, arms, mode, fraction=.5):
    """Pad before the prefix, forecast forward, then score its observed suffix."""
    gains, usable = [], []
    for p in PREFIXES:
        cv, _, anchor, _, deltas = forecast(prefix_input(history, p, mode))
        paths, on = match(cv, anchor, deltas, fraction)
        radius = np.linalg.norm(anchor-cv, axis=-1).max(1)*fraction
        truth = history[:, p:]-history[:, p-1:p]
        n = 8-p
        cv_loss = np.linalg.norm(cv[:, :n]-truth, axis=-1).mean(1)
        err = np.stack([np.linalg.norm(paths[a][:, :n]-truth, axis=-1).mean(1) for a in arms], axis=1)
        values = np.divide(cv_loss[:, None]-err, radius[:, None], out=np.zeros_like(err), where=on[:, None])
        if np.max(np.abs(values), initial=0.) > 1.+1e-7:
            raise ValueError("Prefix gain exceeds the fixed radius")
        gains.append(values)
        usable.append(on)
    g, ok = np.stack(gains), np.stack(usable)
    count = ok.sum(0)
    mean = g.sum(0)/count.clip(1)[:, None]
    return mean, count


def choose(replays, anchor):
    a, na = replays["linear"]
    b, nb = replays["stationary"]
    at = np.arange(len(a))
    pa, pb = a.argmax(1), b.argmax(1)
    single = (na > 0) & (a[at, pa] > a[:, anchor]+1e-10)
    agree = single & (nb > 0) & (pa == pb) & (b[at, pb] > b[:, anchor]+1e-10)
    return {"linear_prefix": np.where(single, pa, anchor),
            "padding_consensus": np.where(agree, pa, anchor)}


def rank_counts(proxy, actual, eligible):
    i, j = np.triu_indices(proxy.shape[1], 1)
    x, y = proxy[eligible], actual[eligible]
    dx, dy = x[:, i]-x[:, j], y[:, i]-y[:, j]
    m = (np.abs(dx) > 1e-10) & (np.abs(dy) > 1e-10)
    return dict(comparable=int(m.sum()), concordant=int(((dx*dy > 0.) & m).sum()), rows=int(eligible.sum()))
