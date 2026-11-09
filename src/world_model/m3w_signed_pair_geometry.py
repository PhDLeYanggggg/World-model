"""Causal, rotation- and agent-order-invariant directional pair features."""

import numpy as np

from src.world_model.m3w_pair_effect import observed_effect


def slope(h, steps):
    t = np.arange(steps, dtype=float)-(steps-1)/2
    return (t @ (h[..., -steps:, :]-h[..., -1:, :])) / (t @ t)


def directional(h, ref, delta, scale):
    scale = np.asarray(scale).reshape(-1)
    rel = h[:, 0]-h[:, 1]
    last = rel[:, -1]
    norm = np.linalg.norm(last, axis=-1)
    axis = last / np.maximum(norm[:, None], 1e-8)
    rel4, rel8 = slope(rel, 4), slope(rel, 8)
    d = delta[:, 0]-delta[:, 1]
    radial = np.einsum('nti,ni->nt', d, axis)
    tangential = d-radial[..., None]*axis[:, None]
    rel_tangent = rel8-np.einsum('ni,ni->n', rel8, axis)[:, None]*axis
    align = (delta[:, 0]*delta[:, 1]).sum(-1).mean(1) / scale**2
    t = np.arange(1, ref.shape[2]+1, dtype=float)[None, None, :, None]
    valid = np.ones(ref.shape[:-1], bool)
    proxy = [observed_effect(ref, delta, p, valid)[0]/scale for p in
             (ref, slope(h, 4)[:, :, None]*t, slope(h, 8)[:, :, None]*t)]
    return np.column_stack((norm/scale, np.sum(rel4*axis, 1)/scale, np.sum(rel8*axis, 1)/scale,
                            np.linalg.norm(rel4-rel8, axis=1)/scale,
                            radial[:, [2, 5, -1]]/scale[:, None],
                            np.sum(tangential[:, -1]*rel_tangent, 1)/scale**2,
                            np.sum(d[:, -1]*rel4, 1)/scale**2, align, *proxy)).astype(np.float32)


WIDTH = 13
