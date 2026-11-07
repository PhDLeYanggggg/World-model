"""Past-backtest-conditioned radius for baseline-relative forecasts."""
import numpy as np


FRACTION = .02
MATERIAL_RADIUS = .25
QUANTILE = .20
BUCKETS = (.03, .08, .20)


def describe(history):
    h = np.asarray(history, float)
    if h.ndim != 3 or h.shape[1:] != (8, 2):
        raise ValueError('Require eight historical positions')
    v = np.diff(h, axis=1)
    scale = 1.+6.5*np.median(np.linalg.norm(v, axis=-1), axis=1)
    error = np.concatenate([np.linalg.norm(h[:, 1+k:]-h[:, 1:-k]-k*v[:, :-k], axis=-1)/k
                            for k in (2, 3)], axis=1)
    bucket = np.digitize(error.mean(1)/scale, BUCKETS)
    return scale, bucket


def quantile(y, source, crowd):
    if not len(y):
        raise ValueError('Require a supported training reference')
    _, inv = np.unique(source, return_inverse=True)
    weight = 1./crowd
    weight /= np.bincount(inv, weights=weight)[inv]
    ix = np.argsort(y, kind='stable')
    w = weight[ix].cumsum()
    return float(y[ix[np.searchsorted(w, QUANTILE*w[-1])]])


def fit(ref, easy, scale, bucket, source, crowd, held):
    use = easy & (source != held)

    def q(mask):
        return quantile(ref[mask]/scale[mask], source[mask], crowd[mask])

    fixed = q(use)
    limits = []
    for k in range(len(BUCKETS)+1):
        at = use & (bucket == k)
        limits.append(q(at) if len(set(source[at])) >= 2 else fixed)
    return fixed, np.asarray(limits)


def clip(cv, bank, radius):
    delta = bank-cv[:, None]
    original = np.linalg.norm(delta, axis=-1).max(2)
    factor = np.minimum(1., np.divide(radius[:, None], original,
                                    out=np.ones_like(original), where=original > 0.))
    result = cv[:, None] + delta*factor[..., None, None]
    return np.where((factor < 1.)[..., None, None], result, bank)
