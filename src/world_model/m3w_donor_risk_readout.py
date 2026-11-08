"""Small donor-only readouts for excess sign, positive size and causal gain."""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge


def design(raw, signed, probes):
    # The last axis indexes predefined model outputs, not raw scene identifiers.
    r = np.concatenate((raw[..., 2:8], signed[..., [2, 3, 4, 5]], probes), axis=-1)
    g = np.concatenate((raw[..., :2], probes), axis=-1)
    return r, g


def _fit(x, y, w, binary=False):
    if not len(y):
        return dict(constant=0.)
    if binary and len(np.unique(y)) < 2:
        return dict(constant=float(y[0]))
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)).clip(1e-6)
    # The total effective weight is fixed across held rosters.
    weight = w/w.sum()*256.
    est = LogisticRegression(C=1., max_iter=500, tol=1e-8) if binary else Ridge(alpha=8.)
    est.fit((x-mean)/scale, y, sample_weight=weight)
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=est.coef_.reshape(-1).tolist(),
                intercept=float(np.asarray(est.intercept_).reshape(-1)[0]), binary=binary)


def predict(x, model):
    if "constant" in model:
        return np.full(x.shape[:-1], model["constant"], dtype=float)
    out = ((x-np.asarray(model["mean"]))/np.asarray(model["scale"]))@np.asarray(model["coef"])+model["intercept"]
    return expit(out) if model["binary"] else out


def fit(sites):
    """Fit on legal actions with equal aggregate weight per donor location."""
    sites = [s for s in sites if len(s[2])]
    r, g, y = (np.concatenate([s[i] for s in sites]) for i in range(3))
    w = np.concatenate([np.full(len(s[2]), 1./len(s[2])) for s in sites])
    risks = []
    for k in (2, 3, 5):
        v = y[:, k]
        pos = v > 0.
        risks.append(dict(sign=_fit(r, pos, w, binary=True), size=_fit(r[pos], v[pos], w[pos]),
                          prior=float(np.average(pos, weights=w))))
    return dict(risk=risks, support=_fit(r, y[:, 9] > 0., w, binary=True),
                gain=[_fit(g, y[:, k], w) for k in (0, 1)])


def apply(readout, r, g):
    p = np.stack([predict(r, m["sign"]) for m in readout["risk"]], axis=-1)
    size = np.stack([predict(r, m["size"]) for m in readout["risk"]], axis=-1).clip(0., 8.)
    gain = np.stack([predict(g, m) for m in readout["gain"]], axis=-1).clip(-8., 8.)
    return dict(prob=p, size=size, gain=gain, support=predict(r, readout["support"]))


def score(gain, penalty, support):
    out = gain.min(axis=-1)-penalty@np.array([2., 4., 1.])+.025*support
    out[:, 0] = 0.
    return out
