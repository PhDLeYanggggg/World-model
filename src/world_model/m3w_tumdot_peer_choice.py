"""Causal same-query peer descriptors at a fixed candidate-action budget."""

import numpy as np

WIDTH = 11


def contexts(history, offsets, bank, incumbent):
    if offsets[0] != 0 or offsets[-1] != len(history) or np.any(np.diff(offsets) < 2):
        raise ValueError("Require a complete partition into multi-agent queries")
    h = np.asarray(history, float)
    scale = np.sqrt(np.square(np.diff(h, axis=1)).sum(2).mean(1)).clip(.05)
    result = np.zeros((*bank.shape[:2], WIDTH), dtype=np.float32)
    shuffle = np.empty(len(h), dtype=int)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        p, z, s = h[lo:hi], bank[lo:hi], scale[lo:hi]
        origin = p[:, -1]
        v = (origin-p[:, -4])/3.
        n = hi-lo
        i = np.arange(n)
        d = origin[None]-origin[:, None]
        ds = np.linalg.norm(d, axis=2)
        np.fill_diagonal(ds, np.inf)
        neighbors = np.argsort(ds, axis=1, kind="stable")[:, :min(3, n-1)]
        j = neighbors[:, 0]
        dv = v[j]-v
        radial = (d[i, j]*dv).sum(1)/ds[i, j].clip(1e-8)
        past_d = np.linalg.norm(p[j]-p, axis=2)/s[:, None]
        past = np.c_[np.log1p(ds[i, j]/s), radial/s, np.linalg.norm(dv, axis=1)/s,
                     np.log1p(past_d.mean(1)), np.log1p(past_d.min(1)), np.full(n, np.log1p(n-1))]
        paths = origin[:, None, None]+z
        peer = paths[:, incumbent][neighbors]
        dist = np.linalg.norm(paths[:, :, None]-peer[:, None], axis=-1)/s[:, None, None, None]
        min_d = dist.min(axis=(2, 3))
        proximity = np.exp(-dist).mean((2, 3))
        final = dist[..., -1].min(2)
        rollout = np.stack((np.log1p(min_d), np.log1p(final), proximity,
                            np.log1p(min_d)-np.log1p(min_d[:, incumbent, None]),
                            proximity-proximity[:, incumbent, None]), axis=-1)
        result[lo:hi] = np.concatenate((np.broadcast_to(past[:, None], (*z.shape[:2], 6)), rollout),
                                       axis=-1).clip(-12., 12.)
        shuffle[lo:hi] = lo+np.roll(i, 1)
    return result, shuffle


def pair_error(path, y, valid, history, offsets):
    """Observed relative-displacement error; no inferred physical safety claim."""
    value, steps = 0., 0
    origin = history[:, -1]
    pred = path+origin[:, None]
    true = y+origin[:, None]
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        i, j = np.triu_indices(hi-lo, 1)
        i, j = i+lo, j+lo
        known = valid[i] & valid[j]
        error = np.linalg.norm((pred[i]-pred[j])-(true[i]-true[j]), axis=-1)
        value += float(error[known].sum())
        steps += int(known.sum())
    return value, steps
