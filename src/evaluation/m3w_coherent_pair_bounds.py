"""Relative-motion bounds that reuse each missing position across its edges."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def assignment_debit(reference_anchors, prediction_anchors):
    """Any bijection is valid; the minimum-distance assignment is no looser."""
    if (reference_anchors.shape != prediction_anchors.shape or reference_anchors.ndim != 2
            or reference_anchors.shape[1] != 2):
        raise ValueError('Expected equally sized 2-D anchor sets')
    if not len(reference_anchors):
        return 0.
    identity = float(np.linalg.norm(reference_anchors-prediction_anchors, axis=-1).sum())
    if len(reference_anchors) == 1 or identity == 0.:
        return identity
    cost = cdist(reference_anchors, prediction_anchors)
    if (cost.diagonal() <= cost.min(axis=1)).all():
        return identity
    i, j = linear_sum_assignment(cost)
    return min(identity, float(cost[i, j].sum()))


def compare(prediction, reference, target, valid):
    if (prediction.shape != reference.shape or target.shape != reference.shape or reference.ndim != 3
            or reference.shape[-1] != 2 or valid.shape != reference.shape[:2] or valid.dtype != bool
            or not np.isfinite(prediction).all() or not np.isfinite(reference).all()
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Require aligned forecasts and masked 2-D targets')
    n, steps = valid.shape
    total = n*(n-1)//2*steps
    prediction, reference, target = (np.asarray(a, dtype=np.float64) for a in (prediction, reference, target))
    diff = prediction-reference
    changed = np.any(diff != 0, axis=(1, 2))
    zero = dict(observed=0., pairwise_debit=0., coherent_debit=0., witness_lower=0., witness_upper=0.,
                informative_stars=0, contracting_stars=0, one_missing_debit=0., two_missing_debit=0.,
                single_anchor_stars=0, near_translation_stars=0, other_multianchor_stars=0)
    if not total or not changed.any():
        return zero
    i, j = np.triu_indices(n, 1)
    active = changed[i] | changed[j]
    i, j = i[active], j[active]
    r, p = reference[i]-reference[j], prediction[i]-prediction[j]
    shift = np.linalg.norm(r-p, axis=-1)
    vi, vj = valid[i], valid[j]
    known = vi & vj
    y = np.where(valid[..., None], target, 0.)
    truth = y[i]-y[j]
    norm = lambda z: np.linalg.norm(z, axis=-1)
    observed = np.where(known, norm(r-truth)-norm(p-truth), 0.).sum()
    one = np.where(vi ^ vj, shift, 0.).sum()
    two = np.where(~(vi | vj), shift, 0.).sum()
    savings = 0.
    informative = contracted = 0
    single = translation = other = 0
    directions = np.zeros_like(reference, dtype=float)
    for t in range(steps):
        obs = np.flatnonzero(valid[:, t])
        if not len(obs):
            continue
        for a in np.flatnonzero(~valid[:, t]):
            keep = obs[np.any(diff[obs, t] != diff[a, t], axis=-1)]
            if not len(keep):
                continue
            old = y[keep, t]+reference[a, t]-reference[keep, t]
            new = y[keep, t]+prediction[a, t]-prediction[keep, t]
            # Equal anchors cancel in the sum; optimize only changed edges.
            debit = float(norm(old-new).sum())
            reduced = max(0., debit-assignment_debit(old, new))
            savings += reduced
            informative += 1
            contracted += int(reduced > 1e-10)
            changes = new-old
            if len(changes) == 1:
                single += 1
            elif np.allclose(changes, changes[0], rtol=0., atol=1e-10):
                translation += 1
            else:
                other += 1
            v = changes.sum(0)
            directions[a, t] = v/max(float(norm(v)), 1e-12)
    coherent = max(0., min(float(one+two), float(one+two-savings)))

    # These are actual completions for one query, not simultaneous assignments
    # for repeated views of the same recording. Time has no extra constraints.
    scale = 16.*(1.+max(float(norm(reference).max()), float(norm(prediction).max()), float(norm(y).max())))
    values = []
    for fill in (reference, prediction, reference+scale*directions, reference-scale*directions):
        completion = np.where(valid[..., None], y, fill)
        true_pair = completion[i]-completion[j]
        values.append((norm(r-true_pair)-norm(p-true_pair)).sum(0))
    witness = np.asarray(values)
    low, high = float(witness.min(0).sum()), float(witness.max(0).sum())
    tolerance = 1e-10*(1.+max(abs(low), abs(high), abs(observed), coherent))
    if low < observed-coherent-tolerance or high > observed+coherent+tolerance:
        raise ValueError(f'A feasible completion escaped the proposed outer bound: '
                         f'{(low, high)} outside {(observed-coherent, observed+coherent)}; '
                         f'dtypes {(prediction.dtype, reference.dtype, target.dtype)}')
    return dict(observed=float(observed/total), pairwise_debit=float((one+two)/total),
                coherent_debit=coherent/total, witness_lower=low/total, witness_upper=high/total,
                informative_stars=informative, contracting_stars=contracted,
                one_missing_debit=float(one/total), two_missing_debit=float(two/total),
                single_anchor_stars=single, near_translation_stars=translation, other_multianchor_stars=other)
