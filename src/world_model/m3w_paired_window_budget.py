"""Reallocate exact peak slots between adjacent windows without using outcomes."""

from itertools import product

import numpy as np
import torch

from src.world_model import m3w_exact_slot_costs as cost


@torch.inference_mode()
def scored_window(net, h, cv, ols, bank, radii):
    """Return the best feasible path and predicted score for each decoder."""
    if not radii:
        return (cv, cv), np.zeros(2)
    if len(radii) > min(2, len(cv)-1):
        return None
    paths, slots = cost.shared.slot_menu(cv, bank, radii)
    options = cost.shared.legal_assignments(slots[0][0], slots[1][0] if len(slots) == 2 else None)
    if not len(options):
        return None
    row = np.concatenate([r for r, _ in slots])
    arm = np.concatenate([c+i*bank.shape[1] for i, (_, c) in enumerate(slots)])
    x, scale, legal, _ = cost.shared.costs.features(h, cv, ols, paths)
    if not legal[row, arm].all():
        raise ValueError('Nonzero slot collapsed numerically')
    v = net(torch.from_numpy(x[row, arm])).numpy().astype(float)*scale[row, None]
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite predicted action costs')
    result, score = [], []
    for s in cost.ranks(v, options):
        best = int(s.argmax())
        take = options[best]
        take = take[take >= 0]
        p = cv.copy()
        p[row[take]] = paths[row[take], arm[take]]
        result.append(p)
        score.append(float(s[best])/len(cv))
    return result, np.asarray(score)


def allocate(left, right, left_radii, right_radii):
    """Score all at most 16 partitions; preserve the original allocation on ties."""
    slots = tuple(left_radii)+tuple(right_radii)
    origin = (0,)*len(left_radii)+(1,)*len(right_radii)
    cache = ({}, {})

    def choices(side, radii):
        if radii not in cache[side]:
            cache[side][radii] = (left, right)[side](radii)
        return cache[side][radii]

    def value(assignment):
        a, b = (choices(i, tuple(r for r, k in zip(slots, assignment, strict=True) if k == i)) for i in (0, 1))
        if a is None or b is None:
            return None
        return [(a[0][i], b[0][i]) for i in (0, 1)], a[1]+b[1]

    fixed = value(origin)
    if fixed is None:
        raise ValueError('Original slot placement is no longer feasible')
    result, scores = list(fixed[0]), fixed[1].copy()
    assigned = [origin, origin]
    feasible = 0
    for allocation in product((0, 1), repeat=len(slots)):
        proposal = value(allocation)
        if proposal is None:
            continue
        feasible += 1
        for i in (0, 1):
            if proposal[1][i] > scores[i]+1e-12:
                result[i], scores[i], assigned[i] = proposal[0][i], proposal[1][i], allocation
    moved = [(sum(a > b for a, b in zip(origin, k, strict=True)),
              sum(a < b for a, b in zip(origin, k, strict=True))) for k in assigned]
    return dict(fixed=fixed[0], pooled=result, moved=moved, feasible_partitions=feasible,
                predicted_gain=(scores-fixed[1]).tolist())
