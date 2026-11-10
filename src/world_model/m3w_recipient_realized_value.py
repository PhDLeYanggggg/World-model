"""Direct observed gain and conservative charges for a nominated swap."""

import numpy as np
from torch import nn

RELATIVE_WEIGHT = .5


def geometry(cv, anchor, drop, add, new, features):
    """Keep only the individual paths and pair edges the swap changes."""
    n, h = cv.shape[:2]
    size = len(drop)
    index = np.arange(n)[None, :]
    other = (index != drop[:, None]) & (index != add[:, None])
    ia = np.concatenate((np.broadcast_to(drop[:, None], (size, n)),
                         np.broadcast_to(add[:, None], (size, n)), drop[:, None]), axis=1)
    ib = np.concatenate((np.broadcast_to(index, (size, n)),
                         np.broadcast_to(index, (size, n)), add[:, None]), axis=1)
    mask = np.column_stack((other, other, np.ones(size, dtype=bool)))
    ind_ref = anchor[np.column_stack((drop, add))]
    ind_new = np.stack((cv[drop], new), axis=1)
    rel_ref = anchor[ia]-anchor[ib]
    rel_new = np.concatenate((cv[drop, None]-anchor[None], new[:, None]-anchor[None],
                              (cv[drop]-new)[:, None]), axis=1)
    delta_ind = np.linalg.norm(ind_new-ind_ref, axis=-1)
    delta_rel = np.where(mask[..., None], np.linalg.norm(rel_new-rel_ref, axis=-1), 0.)
    norm = np.column_stack((delta_ind.sum((1, 2)), delta_rel.sum((1, 2))))
    raw_scale = norm / np.array([n*h, max(1, n*(n-1)//2)*h])
    x = np.column_stack((features, np.log1p(raw_scale), np.log1p(n)*np.ones(size))).astype(np.float32)
    return x, raw_scale, (ind_ref, ind_new, delta_ind, rel_ref, rel_new, delta_rel, ia, ib, mask, norm)


def targets(y, valid, drop, add, info):
    """Benefit plus a nonnegative charge for observed harm and missing steps."""
    r, p, di, rr, rp, dr, i, j, edge, norm = info
    ids = np.column_stack((drop, add))

    def terms(ref, pred, truth, seen, distance, scale):
        d = np.linalg.norm(ref-truth, axis=-1)-np.linalg.norm(pred-truth, axis=-1)
        benefit = np.where(seen, np.maximum(d, 0.), 0.).sum((1, 2))
        charge = np.where(seen, np.maximum(-d, 0.), distance).sum((1, 2))
        return benefit/scale, charge/scale

    a = terms(r, p, y[ids], valid[ids], di, norm[:, 0].clip(1e-12))
    b = terms(rr, rp, y[i]-y[j], valid[i] & valid[j] & edge[..., None], dr, norm[:, 1].clip(1e-12))
    return np.clip(np.column_stack((*a, *b)), 0., 1.).astype(np.float32)


def network(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.SiLU(),
                         nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 4), nn.Sigmoid())


def utility(terms, scale):
    if terms.shape != (len(scale), 4) or scale.shape[1:] != (2,) or not np.isfinite(terms).all():
        raise ValueError('Invalid candidate value prediction')
    trajectory = (terms[:, 0]-terms[:, 1])*scale[:, 0]
    relative = (terms[:, 2]-terms[:, 3])*scale[:, 1]
    return trajectory, trajectory+RELATIVE_WEIGHT*relative
