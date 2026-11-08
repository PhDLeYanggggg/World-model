"""Query-local portfolio selection with identity as an explicit option."""

import numpy as np


def choose_query(cv, ols, excess, legal, cap, risk_slack=0., width=16):
    """Search a diverse shortlist; maximin utility, two simultaneous risk budgets."""
    if not 0 <= cap <= 2:
        raise ValueError("This solver supports zero, one, or two changes per query")
    n, k = cv.shape
    choices = np.full((3, n), -1, dtype=np.int8)
    if not cap or not legal.any():
        return choices
    both = np.minimum(cv, ols)
    excess = excess-risk_slack
    priced = both-4.*np.maximum(excess, 0.).sum(-1)
    scalar = (both, priced, -excess[..., 0], -excess[..., 1])
    legal_ids = np.flatnonzero(legal)
    take = np.unique(np.concatenate([legal_ids[np.argsort(-v.ravel()[legal_ids], kind="stable")[:width]]
                                     for v in scalar]))
    i, a = np.divmod(take, k)
    single = (excess[i, a] <= 0.).all(1)
    score = np.where(single, both[i, a], 0.)
    best = float(max(0., score.max()))
    if best > 0.:
        t = int(score.argmax())
        choices[0, i[t]] = a[t]
    if cap == 2:
        x, y = np.triu_indices(len(take), 1)
        ok = (i[x] != i[y]) & ((excess[i[x], a[x]]+excess[i[y], a[y]]) <= 0.).all(1)
        if ok.any():
            x, y = x[ok], y[ok]
            u = np.minimum(cv[i[x], a[x]]+cv[i[y], a[y]], ols[i[x], a[x]]+ols[i[y], a[y]])
            t = u.argmax()
            if u[t] > best:
                choices[0].fill(-1)
                choices[0, i[[x[t], y[t]]]] = a[[x[t], y[t]]]
    count = int((choices[0] >= 0).sum())
    # These controls see the same entire action bank and copy the joint count only.
    for out, value in zip(choices[1:], (both, priced), strict=True):
        score = np.where(legal, value, -np.inf)
        best = score.argmax(1)
        rows = np.argsort(-score[np.arange(n), best], kind="stable")[:count]
        out[rows] = best[rows]
    return choices


def allocate(cv, ols, excess, legal, inverse, counts, radius, factor):
    """No cross-query ordering or future-dependent capacity."""
    out = np.full((3, len(cv)), -1, dtype=np.int8)
    sizes = np.bincount(inverse, minlength=len(counts))
    if not np.array_equal(inverse, np.repeat(np.arange(len(counts)), sizes)):
        raise ValueError("Require contiguous causal query rows")
    offset = np.r_[0, sizes.cumsum()]
    for q, cap in enumerate(counts):
        if cap:
            s = slice(offset[q], offset[q+1])
            out[:, s] = choose_query(cv[s], ols[s], excess[s], legal[s], int(cap),
                                      risk_slack=factor*radius[s, :, None])
    return out
