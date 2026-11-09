"""Exact shared-menu selection for two interchangeable equal-radius slots."""

import numpy as np
import torch

from src.world_model import m3w_reference_bridge_policy as costs
from src.world_model.m3w_slate_shared_menu import slot_menu


@torch.inference_mode()
def choose(net, history, cv, ols, bank, radius, block=32):
    if len(history) < 3 or block < 1:
        raise ValueError('Require two edited actors and at least one CV actor')
    paths, [(row, arm)] = slot_menu(cv, bank, [radius])
    x, unit, legal, _ = costs.features(history, cv, ols, paths)
    if not legal[row, arm].all():
        raise ValueError('Matched menu lost a legal path')
    if len(np.unique(row)) < 2:
        raise ValueError('No two distinct recipients at this radius')
    center = history[row, -1]
    delta = paths[row, arm] - cv[row]
    scale = unit[row]
    hidden = net.action(torch.from_numpy(x[row, arm]))
    unary = net.unary(hidden).squeeze(-1)
    best = {}
    scored = 0
    for left in range(0, len(row), block):
        first = np.arange(left, min(left+block, len(row)))
        u, b = np.nonzero((first[:, None] < np.arange(len(row))) & (row[first, None] != row[None]))
        if not len(u):
            continue
        a = first[u]
        ha, hb = hidden[a], hidden[b]
        geom = np.column_stack((np.linalg.norm(center[a] - center[b], axis=1),
                                np.linalg.norm(delta[a] - delta[b], axis=-1)[:, [2, 5, 11]]))
        g = torch.from_numpy(np.log1p(geom / ((scale[a] + scale[b]) * .5)[:, None]).astype(np.float32))
        additive = unary[a] + unary[b]
        joint = additive + net.pair(torch.cat((ha + hb, (ha - hb).abs(), g), dim=-1)).squeeze(-1)
        family = arm[a] == arm[b]
        uniform = additive.masked_fill(~torch.from_numpy(family), -torch.inf)
        if not torch.isfinite(joint).all() or not torch.isfinite(additive).all():
            raise ValueError('Nonfinite slate score')
        for name, scores in (('additive', additive), ('joint', joint), ('uniform', uniform)):
            k = int(scores.argmax())
            score = float(scores[k])
            if np.isfinite(score) and (name not in best or score > best[name][0]):
                best[name] = score, (a[k], b[k])
        scored += len(a)
    by_actor = np.bincount(row)
    expected = (len(row)**2 - int(by_actor @ by_actor)) // 2
    if scored != expected:
        raise ValueError('The symmetric search lost a legal assignment')
    out = {}
    for k in ('additive', 'joint', 'uniform'):
        if k not in best:
            out[k] = None
            continue
        score, selection = best[k]
        ix = np.asarray(selection)
        edited = cv.copy()
        edited[row[ix]] = paths[row[ix], arm[ix]]
        peak = np.linalg.norm(edited-cv, axis=-1).max(axis=1)
        np.testing.assert_allclose(np.sort(peak[peak > 0.]), [radius, radius], atol=1e-9, rtol=1e-9)
        out[k] = dict(score=score, path=edited)
    return out, scored
