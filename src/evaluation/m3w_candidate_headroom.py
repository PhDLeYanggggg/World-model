"""Retrospective action-class ceilings, never deployable forecasting policies."""
from __future__ import annotations

import numpy as np


def segment_oracle(baseline, candidate, target, iterations=36):
    """Minimize mean waypoint distance on B + alpha*(N-B), one alpha per path.

    Convex subgradient bisection brackets an optimum. A Lipschitz envelope gives
    a numerical lower loss bound; this is float64, not interval arithmetic or a
    statistical guarantee. Labels deliberately enter this diagnostic only.
    """
    base, cand, y = [np.asarray(v, np.float64) for v in (baseline, candidate, target)]
    if (base.ndim != 3 or base.shape[-1] != 2 or not base.shape[0] or not base.shape[1]
            or cand.shape != base.shape or y.shape != base.shape
            or not all(np.isfinite(v).all() for v in (base, cand, y)) or not 1 <= iterations <= 52):
        raise ValueError('Finite aligned nonempty trajectory labels and fixed iterations required')
    residual, delta = base-y, cand-base
    norm_delta = np.linalg.norm(delta, axis=-1)
    lipschitz = norm_delta.mean(1)
    lo, hi = np.zeros(len(base)), np.ones(len(base))
    for _ in range(iterations):
        mid = (lo+hi)/2
        r = residual+mid[:, None, None]*delta
        norm = np.linalg.norm(r, axis=-1)
        regular = np.divide((r*delta).sum(-1), norm, out=np.zeros_like(norm), where=norm > 0).mean(1)
        singular = np.where(norm == 0, norm_delta, 0).mean(1)
        go_right, go_left = regular+singular < 0, regular-singular > 0
        found = ~(go_right | go_left)
        lo = np.where(go_right | found, mid, lo)
        hi = np.where(go_left | found, mid, hi)
    midpoint = (lo+hi)/2
    costs = np.stack([np.linalg.norm(residual, axis=-1).mean(1),
        np.linalg.norm(cand-y, axis=-1).mean(1),
        np.linalg.norm(residual+midpoint[:, None, None]*delta, axis=-1).mean(1)], axis=1)
    choices = costs.argmin(1)
    feasible = costs[np.arange(len(base)), choices]
    alpha = np.stack([np.zeros(len(base)), np.ones(len(base)), midpoint], axis=1)[np.arange(len(base)), choices]
    slack = 64*np.finfo(np.float64).eps*(1+costs[:, 0]+lipschitz)
    lower = np.maximum(0., feasible-lipschitz*(hi-lo)-slack)
    return dict(alpha=alpha, feasible_loss=feasible, lower_loss=lower,
                binary_loss=costs[:, :2].min(1), reference_loss=costs[:, 0],
                bracket_width=hi-lo, lipschitz=lipschitz)


def budget_envelope(reference, feasible_loss, lower_loss, scenes, fraction):
    """Relax per-query intervention budgets into optimistic per-scene counts.

    Ceiling uses labels, picks the most beneficial rows and ignores pairwise
    compatibility. It is not a learned risk-coverage curve or physical safety.
    """
    ref, feasible, lower = [np.asarray(v, np.float64) for v in (reference, feasible_loss, lower_loss)]
    scenes = np.asarray(scenes)
    if (ref.ndim != 1 or not len(ref) or any(v.shape != ref.shape for v in (feasible, lower, scenes))
            or not all(np.isfinite(v).all() for v in (ref, feasible, lower))
            or np.any(lower > feasible+1e-10) or not 0 <= fraction <= 1):
        raise ValueError('Aligned finite losses and a unit-interval budget required')
    rows, results = 0, []
    for scene in np.unique(scenes):
        use = scenes == scene
        n = int(use.sum())
        count = int(np.ceil(fraction*n))
        gain = np.sort(np.maximum(ref[use]-feasible[use], 0))[::-1]
        optimistic = np.sort(np.maximum(ref[use]-lower[use], 0))[::-1]
        results.append(dict(scene=str(scene), rows=n, allowed_rows=count,
            reference_ADE=float(ref[use].mean()),
            feasible_gain_ADE=float(gain[:count].sum()/n),
            optimistic_gain_ADE=float(optimistic[:count].sum()/n)))
        rows += count
    base = float(np.mean([r['reference_ADE'] for r in results]))
    a = float(np.mean([r['feasible_gain_ADE'] for r in results]))
    b = float(np.mean([r['optimistic_gain_ADE'] for r in results]))
    return dict(fraction=fraction, allowed_rows=rows, reference_ADE=base,
        feasible_gain_ADE=a, optimistic_gain_ADE=b,
        feasible_gain_percent=100*a/base if base > 0 else None,
        optimistic_gain_percent=100*b/base if base > 0 else None, scenes=results,
        result_kind='future_label_oracle_not_inference', local_query_budgets_relaxed=True)
