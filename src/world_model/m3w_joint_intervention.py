"""Baseline-relative intervention primitives; no fitted model or safety certificate.

Inference consumes predicted scores and candidate rollouts, never target labels.
Risk tolerances and coordinate-distance thresholds have no experiment defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


@dataclass
class InterventionProblem:
    expected_gain: np.ndarray
    expected_harm: np.ndarray
    supported: np.ndarray
    edges: np.ndarray
    pair_cost: np.ndarray
    pair_weight: float
    max_mean_predicted_harm: float
    max_interventions: int

    def validate(self) -> None:
        n = len(self.expected_gain)
        if n == 0 or any(np.shape(v) != (n,) for v in
                         (self.expected_gain, self.expected_harm, self.supported)):
            raise ValueError("Nonempty aligned agent score vectors required")
        if np.asarray(self.supported).dtype != bool:
            raise ValueError("Support must be an explicit Boolean mask")
        if not np.isfinite(self.expected_gain).all() or not np.isfinite(self.expected_harm).all():
            raise ValueError("Nonfinite scores cannot be certified or optimized")
        if np.any(self.expected_harm < 0):
            raise ValueError("Expected positive harm cannot be negative")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2 or self.edges.dtype.kind not in "iu":
            raise ValueError("Edges must be integer [edge, 2] indices")
        if len(self.edges) and (np.any(self.edges < 0) or np.any(self.edges >= n) or
                              np.any(self.edges[:, 0] >= self.edges[:, 1]) or
                              len(np.unique(self.edges, axis=0)) != len(self.edges)):
            raise ValueError("Edges must be unique ordered agent pairs")
        if self.pair_cost.shape != (len(self.edges), 2, 2) or not np.isfinite(self.pair_cost).all():
            raise ValueError("Pair costs must be finite [edge, 2, 2]")
        if np.any(self.pair_cost < 0) or np.any(self.pair_cost[:, 0, 0] != 0):
            raise ValueError("Use nonnegative excess pair cost with zero floor cost")
        for value in (self.pair_weight, self.max_mean_predicted_harm):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Explicit finite nonnegative weights and budgets required")
        if not isinstance(self.max_interventions, (int, np.integer)) or not 0 <= self.max_interventions <= n:
            raise ValueError("Intervention cap must be an integer between zero and agent count")


def past_proximity_edges(current_xy: np.ndarray, *, radius: float) -> np.ndarray:
    xy = np.asarray(current_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Finite current positions in a common coordinate frame required")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Explicit positive dataset-local graph radius required")
    i, j = np.triu_indices(len(xy), k=1)
    selected = np.linalg.norm(xy[i] - xy[j], axis=1) <= radius
    return np.column_stack([i[selected], j[selected]]).astype(np.int64)


def proximity_cost_table(baseline: np.ndarray, candidate: np.ndarray, edges: np.ndarray,
                         *, distance_threshold: float) -> np.ndarray:
    b, c = np.asarray(baseline), np.asarray(candidate)
    if b.shape != c.shape or b.ndim != 3 or b.shape[-1] != 2 or b.shape[1] == 0:
        raise ValueError("Aligned [agent, future-step, xy] rollouts required")
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("Candidate rollouts must be finite; handle unsupported forecasts first")
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("Explicit positive common-coordinate distance threshold required")
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu" or np.any(edges < 0) or np.any(edges >= len(b)):
        raise ValueError("Invalid edge indices")
    costs = np.zeros((len(edges), 2, 2))
    i, j = edges[:, 0], edges[:, 1]
    for ai, left in enumerate((b, c)):
        for aj, right in enumerate((b, c)):
            d = np.linalg.norm(left[i] - right[j], axis=-1)
            costs[:, ai, aj] = np.square(np.maximum(0, 1 - d / distance_threshold)).mean(axis=1)
    return np.maximum(0, costs - costs[:, :1, :1])


def select_interventions(problem: InterventionProblem, *, mode: str,
                         time_limit_seconds: float = 30.) -> dict:
    """Binary control arms with fixed forecasts; caps do not imply matched realized coverage."""
    problem.validate()
    if mode not in {"floor", "uncontrolled", "independent", "scene_uniform", "joint"}:
        raise ValueError("Unknown intervention control arm")
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("Positive solver time limit required")
    p, n = problem, len(problem.expected_gain)
    # E[max(-gain, 0)] >= max(-E[gain], 0); reconcile separately predicted heads.
    coherent_harm = np.maximum(p.expected_harm, -p.expected_gain)
    x = np.zeros(n, dtype=bool)
    reason, optimal = "floor", False

    def describe(bits):
        pairs = p.pair_cost[np.arange(len(p.edges)), bits[p.edges[:, 0]].astype(int), bits[p.edges[:, 1]].astype(int)]
        pair = float(pairs.mean()) if len(pairs) else 0.
        gain = float(np.mean(bits * p.expected_gain))
        harm = float(np.mean(bits * coherent_harm))
        return gain, harm, pair, -gain + p.pair_weight * pair

    def feasible(bits):
        return (not np.any(bits & ~p.supported) and bits.sum() <= p.max_interventions and
                np.mean(bits * coherent_harm) <= p.max_mean_predicted_harm + 1e-10)

    if mode == "uncontrolled":
        x, reason = p.supported.copy(), "uncontrolled_supported_candidates"
    elif mode == "scene_uniform":
        whole = np.ones(n, dtype=bool)
        if feasible(whole) and describe(whole)[-1] < 0:
            x, reason = whole, "selected"
        optimal = True
    elif mode in {"independent", "joint"}:
        edges = p.edges if mode == "joint" else np.empty((0, 2), dtype=int)
        m = len(edges)
        objective = np.r_[-p.expected_gain / n, np.zeros(m)]
        rows, cols, values = [], [], []
        upper = [p.max_mean_predicted_harm, float(p.max_interventions)]
        for i in range(n):
            rows.extend([0, 1]); cols.extend([i, i]); values.extend([coherent_harm[i]/n, 1.])
        # y_ij = x_i*x_j via the three binary-product linear constraints.
        for e, (i, j) in enumerate(edges):
            cost = p.pair_cost[e] * p.pair_weight / m
            objective[i] += cost[1, 0]
            objective[j] += cost[0, 1]
            objective[n+e] = cost[1, 1] - cost[1, 0] - cost[0, 1]
            for coefficients, cap in [({n+e: 1., i: -1.}, 0.),
                                      ({n+e: 1., j: -1.}, 0.),
                                      ({i: 1., j: 1., n+e: -1.}, 1.)]:
                row = len(upper)
                for column, value in coefficients.items():
                    rows.append(row); cols.append(column); values.append(value)
                upper.append(cap)
        matrix = csc_matrix((values, (rows, cols)), shape=(len(upper), n+m))
        result = milp(c=objective, integrality=np.r_[np.ones(n), np.zeros(m)],
                      bounds=Bounds(np.zeros(n+m), np.r_[p.supported.astype(float), np.ones(m)]),
                      constraints=LinearConstraint(matrix, np.full(len(upper), -np.inf), np.array(upper)),
                      options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.})
        if not result.success:
            reason = "solver_not_optimal_floor"
        else:
            optimal = True
            bits = result.x[:n] > .5
            selection_value = describe(bits)[-1] if mode == "joint" else -describe(bits)[0]
            if (np.allclose(result.x[:n], bits, atol=1e-7, rtol=0) and feasible(bits) and selection_value < 0):
                x, reason = bits, "selected"
            else:
                reason = "no_feasible_positive_gain_floor"
    gain, harm, pair, objective = describe(x)
    return {"switch": x, "mode": mode, "reason": reason, "solver_optimal": optimal,
            "mean_predicted_gain": gain, "mean_predicted_harm": harm,
            "mean_raw_predicted_harm": float(np.mean(x*p.expected_harm)),
            "mean_pair_proxy": pair, "objective": objective, "switch_rate": float(x.mean()),
            "predicted_constraints_satisfied": bool(feasible(x)),
            "realized_risk_certified": False, "physical_safety_certified": False}


def realized_relative_costs(baseline, candidate, targets, valid_mask, past_scales) -> dict:
    """Loss/evaluation labels only. Missing endpoints never become last-valid FDE."""
    b, c, y = (np.asarray(a, dtype=float) for a in (baseline, candidate, targets))
    mask, scale = np.asarray(valid_mask), np.asarray(past_scales, dtype=float)
    if b.shape != c.shape or b.shape != y.shape or b.ndim != 3 or b.shape[-1] != 2 or b.shape[1] == 0:
        raise ValueError("Aligned rollout and target arrays required")
    if mask.shape != b.shape[:2] or mask.dtype != bool or scale.shape != (len(b),):
        raise ValueError("Boolean loss mask and past-only per-agent scales required")
    if not np.isfinite(b).all() or not np.isfinite(c).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Nonfinite predictions or valid labels")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Finite positive past scales required")
    errors = [np.linalg.norm(pred-y, axis=-1) / scale[:, None] for pred in (b, c)]
    count = mask.sum(axis=1)
    ade = [np.divide(np.where(mask, err, 0.).sum(axis=1), count,
                     out=np.full(len(b), np.nan), where=count > 0) for err in errors]
    fde = [np.where(mask[:, -1], err[:, -1], np.nan) for err in errors]
    return {"baseline_ade": ade[0], "candidate_ade": ade[1],
            "gain_ade": ade[0]-ade[1], "harm_ade": np.maximum(ade[1]-ade[0], 0),
            "gain_fde": fde[0]-fde[1], "harm_fde": np.maximum(fde[1]-fde[0], 0),
            "ade_valid": count > 0, "fde_valid": mask[:, -1].copy(),
            "data_role": "loss_or_evaluation_labels_only"}


def screen_cluster_risks(*, losses, cluster_ids, fitted_cluster_ids, policy_ids,
                         lower, upper, tolerance, delta: float) -> dict:
    """Hoeffding/union-bound diagnostic for a fixed policy family, not novel theory.

    One row must already aggregate one independent cluster. IDs cannot verify IID,
    fixed-before-calibration policies, legal bounds, or correct cluster membership.
    """
    loss = np.asarray(losses, dtype=float)
    lo, hi, tol = (np.asarray(v, dtype=float) for v in (lower, upper, tolerance))
    if loss.ndim != 3 or any(d < 1 for d in loss.shape):
        raise ValueError("Nonempty [independent-cluster, frozen-policy, risk] losses required")
    n, m, k = loss.shape
    if len(cluster_ids) != n or len(set(cluster_ids)) != n:
        raise ValueError("Each calibration cluster must be unique; overlapping windows are not clusters")
    if set(cluster_ids) & set(fitted_cluster_ids):
        raise ValueError("Calibration clusters overlap fitted/development clusters")
    if len(policy_ids) != m or len(set(policy_ids)) != m:
        raise ValueError("Unique frozen policy identities required")
    if any(v.shape != (k,) or not np.isfinite(v).all() for v in (lo, hi, tol)):
        raise ValueError("Explicit finite risk bounds/tolerances required")
    if np.any(hi <= lo) or np.any(tol < lo) or np.any(tol > hi) or not 0 < delta < 1:
        raise ValueError("Invalid bounds, tolerance or error probability")
    if not np.isfinite(loss).all() or np.any(loss < lo) or np.any(loss > hi):
        raise ValueError("Observed losses violate the declared bounds")
    radius = (hi-lo) * np.sqrt(np.log(m*k/delta)/(2*n))
    bound = np.minimum(hi, loss.mean(axis=0) + radius)
    return {"upper_risk_bound": bound, "accepted": (bound <= tol).all(axis=1),
            "independent_cluster_count_declared": n, "family_size": m, "risk_count": k,
            "independence_verified": False, "fixed_family_verified": False,
            "physical_safety_certified": False,
            "scope": "bounded_cluster_mean_under_declared_assumptions_only"}
