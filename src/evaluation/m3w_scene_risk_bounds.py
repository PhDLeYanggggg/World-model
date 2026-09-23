"""Alternative bounded-risk diagnostic, NOT a replacement calibration protocol.

Hoeffding-Bentkus follows Learn then Test, arXiv:2110.01052v5, Proposition 1.
This is an existing statistical result, not a new M3W theorem. No data role,
independence, risk definition, model improvement or deployment is established.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from src.world_model.m3w_joint_intervention import screen_cluster_risks


def _check(mean, n, probability):
    if (not math.isfinite(mean) or not 0 <= mean <= 1
            or type(n) is not int or n < 1
            or not math.isfinite(probability) or not 0 < probability < 1):
        raise ValueError("Finite unit-interval mean, positive integer scene count and probability required")


def hb_p_value(observed_mean, n, null_mean):
    """Lower-tail test of a bounded mean exceeding null_mean, under IID units."""
    _check(observed_mean, n, null_mean)
    if observed_mean >= null_mean:
        return 1.
    if observed_mean == 0:
        divergence = -math.log1p(-null_mean)
    else:
        divergence = (observed_mean * math.log(observed_mean / null_mean)
                      + (1-observed_mean) * math.log((1-observed_mean)/(1-null_mean)))
    log_chernoff = -n * max(divergence, 0.)
    # Fractional bounded scene losses are not Bernoulli counts. The theorem
    # requires ceil(n * mean); floor or an uninflated binomial CDF is invalid.
    log_bentkus = 1. + float(binom.logcdf(math.ceil(n * observed_mean), n, null_mean))
    return math.exp(min(0., log_chernoff, log_bentkus))


def hb_upper_mean(observed_mean, n, error_level):
    """Invert the fixed-mean test; return the upper side of the numerical bracket."""
    _check(observed_mean, n, error_level)
    if observed_mean == 1:
        return 1.
    if observed_mean == 0:
        return -math.expm1(math.log(error_level) / n)
    low, high = float(observed_mean), 1.
    for _ in range(80):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        if hb_p_value(observed_mean, n, middle) <= error_level:
            high = middle
        else:
            low = middle
    return high


def zero_loss_required_scenes(tolerance, delta, *, family_size, risk_count):
    """Optimistic arithmetic only; tolerance is not the 2% easy-ADE criterion."""
    if (type(family_size) is not int or family_size < 1
            or type(risk_count) is not int or risk_count < 1):
        raise ValueError("Positive integer family and risk counts required")
    _check(0., 1, tolerance)
    _check(0., 1, delta)
    level = delta / (family_size * risk_count)
    n = max(1, math.ceil(math.log(level) / math.log1p(-tolerance)))
    while hb_upper_mean(0., n, level) > tolerance:
        n += 1
    return n


def screen_hb_diagnostic(*, losses, cluster_ids, fitted_cluster_ids, policy_ids,
                         lower, upper, tolerance, delta):
    """Reuse existing shape/exposure checks; preserve the frozen legacy screen.

IDs cannot prove physical-site independence, fixed families, population match or
legal bounds. This entry is not wired into training/calibration/confirmation.
"""
    reference = screen_cluster_risks(losses=losses, cluster_ids=cluster_ids,
        fitted_cluster_ids=fitted_cluster_ids, policy_ids=policy_ids,
        lower=lower, upper=upper, tolerance=tolerance, delta=delta)
    loss = np.asarray(losses, dtype=float)
    lo, hi, tol = (np.asarray(x, dtype=float) for x in (lower, upper, tolerance))
    n, m, k = loss.shape
    level = delta / (m * k)
    normalized = (loss - lo) / (hi - lo)
    bound = np.empty((m, k))
    for j in range(m):
        for r in range(k):
            mean = math.fsum(normalized[:, j, r]) / n
            bound[j, r] = lo[r] + (hi[r]-lo[r]) * hb_upper_mean(mean, n, level)
    return {
        "upper_risk_bound": bound, "accepted": (bound <= tol).all(axis=1),
        "hoeffding_reference_bound": reference["upper_risk_bound"],
        "independent_cluster_count_declared": n, "family_size": m, "risk_count": k,
        "per_hypothesis_error_level": level,
        "method": "hoeffding_bentkus_bonferroni_diagnostic",
        "independence_verified": False, "fixed_family_verified": False,
        "physical_safety_certified": False, "real_calibration_executed": False,
        "risk_definition_or_protocol_changed": False,
        "scope": "bounded_cluster_mean_under_declared_assumptions_only",
    }


def joint_monotonicity_witness():
    """Constructed endpoint-proximity proxy, not a physical collision experiment."""
    baseline = np.array([[0., 0.], [4., 0.]])
    candidate = np.array([[4., 0.], [0., 0.]])
    masks = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    events, distances = [], []
    for mask in masks:
        selected = np.where(mask[:, None], candidate, baseline)
        distance = float(np.linalg.norm(selected[0]-selected[1]))
        distances.append(distance)
        events.append(int(distance < 1.))
    return {
        "scope": "synthetic_two_agent_one_step_endpoint_proxy",
        "baseline": baseline.tolist(), "candidate": candidate.tolist(),
        "switch_masks": masks.tolist(), "switch_counts": masks.sum(axis=1).tolist(),
        "pair_distances": distances, "proximity_events": events,
        "monotone_in_decreasing_intervention": all(b <= a for a, b in zip(events, events[1:])),
        "physical_collision_claim": False, "measured_m3w_failure": False,
    }
