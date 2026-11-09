"""Ablate only the pair score on a shared fixed-slot action menu."""

import numpy as np
import torch

from src.world_model import m3w_reference_bridge_policy as costs
from src.world_model.m3w_slate_imitation import pack


def slot_menu(cv, bank, radii):
    if not 1 <= len(radii) <= min(2, len(cv)-1) or not np.all(np.asarray(radii) > 0):
        raise ValueError('Require one or two positive peak slots and one CV recipient')
    offset = bank - cv[:, None]
    peak = np.linalg.norm(offset, axis=-1).max(axis=-1)
    def slot(radius):
        valid = (peak > 0) & (peak >= radius - 1e-10)
        fraction = np.divide(radius, peak, out=np.zeros_like(peak), where=valid)
        return offset * np.minimum(1., fraction)[..., None, None], np.nonzero(valid)

    deltas, slots = zip(*(slot(r) for r in radii), strict=True)
    return cv[:, None] + np.concatenate(deltas, axis=1), slots


def legal_assignments(first, second=None):
    if second is None:
        return np.column_stack((np.arange(len(first)), np.full(len(first), -1)))
    i, j = np.nonzero(first[:, None] != second)
    return np.column_stack((i, j+len(first)))


@torch.inference_mode()
def choose(net, h, cv, ols, bank, radii):
    paths, slots = slot_menu(cv, bank, radii)
    opts = legal_assignments(slots[0][0], slots[1][0] if len(slots) == 2 else None)
    if not len(opts):
        raise ValueError('Frozen peaks have no feasible assignment in the common bank')
    x, scale, legal, _ = costs.features(h, cv, ols, paths)
    row = np.concatenate([a[0] for a in slots])
    arm = np.concatenate([a[1]+j*bank.shape[1] for j, a in enumerate(slots)])
    if not legal[row, arm].all():
        raise ValueError('Action at a positive peak collapsed numerically')
    delta = paths[row, arm] - cv[row]
    peak = np.linalg.norm(delta, axis=-1).max(axis=-1)
    np.testing.assert_allclose(peak, np.repeat(radii, [len(a[0]) for a in slots]), atol=1e-9, rtol=1e-9)
    best = [(-np.inf, None), (-np.inf, None)]
    for at in range(0, len(opts), 8192):
        batch = opts[at:at+8192]
        inputs = pack(x[row, arm], h[row, -1], delta, scale[row], batch)
        for i, scores in enumerate(net(*(torch.from_numpy(v) for v in inputs))):
            k = int(scores.argmax())
            value = float(scores[k])
            if not np.isfinite(value):
                raise ValueError('Nonfinite shared-menu score')
            if value > best[i][0]:
                best[i] = value, batch[k]
    predictions = []
    for _, selected in best:
        selected = selected[selected >= 0]
        p = cv.copy()
        p[row[selected]] = paths[row[selected], arm[selected]]
        predictions.append(p)
    return predictions[0], predictions[1], len(opts)
