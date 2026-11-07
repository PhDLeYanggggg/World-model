"""Shared label witnesses with a motion-extrapolated terminal center."""
import numpy as np

from src.evaluation.m3w_completion_occlusion import corridor
from src.evaluation.m3w_pair_completion_witness import completion_ledger, extreme
from src.evaluation.m3w_joint_path_audit import local_edges, query


def recenter(known, base, radius, gaps, n_tracks):
    """Keep the old radius and internal anchors; move eligible terminal centers."""
    if n_tracks <= 0 or len(known) % n_tracks:
        raise ValueError('Inconsistent ledger track index')
    length = len(known)//n_tracks
    coords = base.copy()
    counts = dict(terminal_steps=0, shifted_eligible_steps=0, changed_coordinates=0)
    for k in range(n_tracks):
        sl = slice(k*length, (k+1)*length)
        seen = known[sl]
        obs = np.flatnonzero(seen)
        ix = np.flatnonzero((gaps[sl] >= 0) & (np.arange(length) > obs[-1]))
        counts['terminal_steps'] += len(ix)
        if not len(ix) or len(obs) < 4 or not (np.diff(obs[-4:]) == 1).all():
            continue
        p, r = corridor(base[sl], seen, ix, 'ols4')
        idx = k*length+ix
        np.testing.assert_array_equal(r, radius[idx])
        counts['shifted_eligible_steps'] += len(ix)
        counts['changed_coordinates'] += int(np.any(p != coords[idx], axis=1).sum())
        coords[idx] = p
    np.testing.assert_array_equal(coords[known], base[known])
    return coords, counts


def recording(agents, frames, history, target, valid, paths, queries):
    ids, known, old, radius, gaps = completion_ledger(agents, frames, history, target, valid)
    center, counts = recenter(known, old, radius, gaps, len(np.unique(agents)))
    terms = []
    observed = np.zeros(len(paths))
    checked = np.zeros_like(observed)
    n = 0
    for ix in queries:
        l, r = local_edges(history[ix]); a, b = ix[l], ix[r]
        if not len(a):
            continue
        n += 1
        both = valid[a] & valid[b]
        rel = history[a, -1]-history[b, -1]
        y = target[a][both]-target[b][both]
        for s, p in enumerate(paths):
            u, v = p[:, a]-p[:, b]+rel[None, :, None]
            observed[s] += (np.linalg.norm(u[both]-y, axis=1)-np.linalg.norm(v[both]-y, axis=1)).sum()/both.size
            change = ~both & np.any(u != v, axis=2)
            if change.any():
                terms.append((s, np.column_stack((ids[a][change], ids[b][change])), u[change], v[change], 1./both.size))
            _, score, _, _ = query(history[ix], p[:, ix], center[ids[ix]]-history[ix, -1, None],
                                   np.ones_like(valid[ix]), [(0, 1)])
            checked[s] += score[0, 1]

    def measured(x):
        out = observed.copy()
        for s, e, u, v, w in terms:
            rel = x[e[:, 0]]-x[e[:, 1]]
            out[s] += w*(np.linalg.norm(u-rel, axis=1)-np.linalg.norm(v-rel, axis=1)).sum()
        return out

    values = [measured(center)]
    np.testing.assert_allclose(checked, values[0], rtol=1e-10, atol=1e-8)
    if not terms:
        return n, np.asarray(values*5+[observed]), counts
    edge, left, right = (np.concatenate([z[j] for z in terms]) for j in (1, 2, 3))
    w = np.concatenate([np.full(len(e), a/len(paths)) for _, e, _, _, a in terms])
    for scale in (1., 2.):
        for sign in (-1., 1.):
            effect, x, _ = extreme(center, radius*scale, gaps, edge, left, right, w, sign)
            np.testing.assert_array_equal(x[known], old[known])
            if (np.abs(x-center).max(axis=1) > radius*scale+1e-9).any():
                raise ValueError('Motion-centered witness exceeds registered coordinate budget')
            values.append(measured(x))
            np.testing.assert_allclose(values[-1].mean()-observed.mean(), effect, rtol=1e-10, atol=1e-8)
    return n, np.asarray(values+[observed]), counts
