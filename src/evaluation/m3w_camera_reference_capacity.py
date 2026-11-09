"""Camera-input reference controls and radius-limited hindsight bounds."""
import numpy as np

from src.evaluation.m3w_cv_anchor_budget import candidate_terms

NAMES = ('ols4_anchored', 'ols4_fitted', 'median4_anchored', 'median4_fitted')


def linear_menu(history, horizon):
    """Separate velocity smoothing from de-noising the final observed position."""
    past = np.asarray(history, dtype=float)[:, -4:]
    if past.shape[1:] != (4, 2) or not np.isfinite(past).all():
        raise ValueError('Require at least four finite history points')
    past = past - past[:, -1:]
    x = np.arange(-3., 1.)
    i, j = np.triu_indices(4, k=1)
    ols = np.einsum('t,ntd->nd', x - x.mean(), past) / 5.
    med = np.median((past[:, j] - past[:, i]) / (j-i)[None, :, None], axis=1)
    a = past.mean(1) - x.mean() * ols
    b = np.median(past - x[None, :, None] * med[:, None], axis=1)
    t = np.arange(1., horizon + 1)[None, :, None]
    return np.stack((ols[:, None] * t, a[:, None] + ols[:, None] * t,
                     med[:, None] * t, b[:, None] + med[:, None] * t), axis=1)


def matched_menu(cv, paths, eligible, radius):
    """Shorten available displacements; never make a direction larger."""
    delta = paths - cv[:, None]
    peak = np.linalg.norm(delta, axis=-1).max(-1)
    legal = eligible[:, None] & (radius[:, None] > 0.) & (peak >= radius[:, None])
    w = np.divide(radius[:, None], peak, out=np.zeros_like(peak), where=legal)
    return cv[:, None] + w[..., None, None] * delta, legal


def past_rank(history, cv, camera_paths, legal, query, counts):
    """Nominate with rolling-origin errors wholly inside the observed prefix."""
    score = np.zeros(legal.shape)
    for end in range(4, history.shape[1]):
        h = history[:, :end]
        n = min(3, history.shape[1]-end)
        future = history[:, end:end+n] - h[:, -1:]
        t = np.arange(1., n+1)[None, :, None]
        reference = (h[:, -1:] - h[:, -2:-1]) * t
        fit = linear_menu(h, n)
        score += np.linalg.norm(reference-future, axis=-1).mean(1)[:, None]
        score -= np.linalg.norm(fit-future[:, None], axis=-1).mean(2)
    score = np.where(legal, score, -np.inf)
    arm = score.argmax(1)
    gain = score[np.arange(len(arm)), arm]
    path = cv.copy()
    for q, count in enumerate(counts):
        rows = np.flatnonzero((query == q) & np.isfinite(gain))
        if len(rows) < count:
            raise ValueError('Past ranker cannot reproduce the fixed count')
        use = rows[np.argsort(-gain[rows], kind='stable')[:count]]
        path[use] = camera_paths[use, arm[use]]
    return path


def radius_bounds(cv, ols, y, valid, query, eligible, radius, counts):
    """Separate upper bounds against CV/OLS4 at each fixed query count.

    For a known step the maximum possible reduction in CV error is the
    smaller of radius and CV error. For a missing step, the reduction of
    OLS4 disagreement is at most the smaller of radius and disagreement.
    The two bounds may favor different recipients and ignore risk.
    """
    _, size = np.unique(query, return_counts=True)
    weights = 1. / size[query]
    mass, base, _, _ = candidate_terms(cv, ols, cv[:, None], y, valid)
    ref = float(weights @ mass)
    if ref <= 0.:
        return None
    ce = np.linalg.norm(cv-np.where(valid[..., None], y, 0.), axis=-1)
    oe = np.linalg.norm(ols-np.where(valid[..., None], y, 0.), axis=-1)
    known = np.where(valid, np.minimum(ce, radius[:, None]), 0.).sum(1)
    unseen = np.where(valid, 0., np.minimum(np.linalg.norm(ols-cv, axis=-1), radius[:, None])).sum(1)

    def limit(a):
        return sum(np.sort((weights*a)[(query == q) & eligible])[::-1][:n].sum()
                   for q, n in enumerate(counts))

    return dict(known_pp=(100./ref * np.array([limit(known), limit(known)+weights@np.where(valid, oe-ce, 0.).sum(1)])).tolist(),
                conservative_pp=(100./ref * np.array([limit(known), limit(known+unseen)+weights@base])).tolist())
