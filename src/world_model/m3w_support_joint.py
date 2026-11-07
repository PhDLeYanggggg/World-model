"""Equal-budget easy-reference retrieval and ordered causal joint edits."""
import numpy as np

CUTS = ((.25, 0.), (.25, .1), (.25, .2), (.25, .35),
        (1., 0.), (1., .1), (1., .2), (1., .35))
ARMS = ('family_joint', 'support_insert', 'support_joint')


def retrieve(legal, credit, ref, utility, confidence):
    """Exactly eight slots when available; no row or family quota."""
    valid = np.flatnonzero(legal.ravel())
    orders = [valid[np.argsort(-v.ravel()[valid], kind='stable')]
              for v in (credit, ref, confidence, utility)]
    chosen = {}
    for slot in range(min(8, len(valid))):
        ix = next(i for i in orders[slot % len(orders)] if i not in chosen)
        chosen[ix] = None
    return np.asarray(list(chosen), dtype=int)


def select(direct, queries, estimate, removal, arm, cuts):
    """No observed outcomes, support from future queries, or future totals."""
    if arm not in ARMS:
        raise ValueError('Unknown joint arm')
    out = direct.copy()
    if cuts is None:
        return out
    limit, confidence = cuts
    u, a, e, ref, joint = estimate
    rg, ra, re, rr = removal.T
    added, seen = False, 0.
    for q, family, support in queries:
        old = q[direct[q] >= 0]
        seen += rr[old].sum()
        deletable = old[(rg[old] >= 0.) & (ra[old] >= 0.) & (re[old] >= 0.) & (rr[old] <= .05)]
        if added:
            if arm != 'support_insert' and len(deletable):
                i = deletable[(rg[deletable]+ra[deletable]+re[deletable]).argmax()]
                if ra[i]+re[i] > 0.:
                    out[i] = -1
            continue
        if seen > limit:
            continue
        take = family if arm == 'family_joint' else support
        best = None
        for i, f in take:
            if joint[i, f] <= confidence or ref[i, f] <= 0.:
                continue
            # A same-query pair can make room without moving any old switch.
            options = ([-1] if len(old) < min(2, len(q)-1) else [])
            if arm != 'support_insert':
                options += deletable.tolist()
            for remove in options:
                dg, da, de = (0., 0., 0.) if remove < 0 else (rg[remove], ra[remove], re[remove])
                gain = u[i, f]+dg
                if gain <= 0. or a[i, f] > da or e[i, f] > de:
                    continue
                key = (gain, joint[i, f], -int(remove >= 0), -int(i), -int(f), -int(remove))
                if best is None or key > best[0]:
                    best = key, i, f, remove
        if best is not None:
            _, i, f, remove = best
            out[i] = f
            if remove >= 0:
                out[remove] = -1
            added = True
    return out
