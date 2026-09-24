"""Query-local credit/denominator controls on frozen expected net easy costs."""
import math
import itertools
import numpy as np
from src.world_model.m3w_net_easy_guard import allocate

MODES = ('net_population', 'net_clipped_population', 'net_selected', 'net_clipped_selected')


def coefficients(net, denominator, mode, rho=.02):
    q, r = np.asarray(net, dtype=float), np.asarray(denominator, dtype=float)
    if (q.ndim != 1 or r.shape != q.shape or not np.isfinite(q).all()
            or not np.isfinite(r).all() or np.any(r < 0) or mode not in MODES or rho != .02):
        raise ValueError('Finite query-local expected costs, nonnegative denominator and fixed rho required')
    raw = np.maximum(q, 0) if 'clipped' in mode else q.copy()
    if mode.endswith('_selected'):
        return raw-rho*r, 0.
    return raw, rho*math.fsum(r)


def direct_risk(net, denominator, bits, mode, rho=.02):
    coefficients(net, denominator, mode, rho)
    b = np.asarray(bits)
    if b.dtype != bool or b.shape != np.asarray(net).shape:
        raise ValueError('Aligned boolean selection required')
    raw = np.maximum(net, 0) if 'clipped' in mode else np.asarray(net)
    risk = math.fsum(raw[b])
    budget = rho*math.fsum(np.asarray(denominator)[b] if mode.endswith('_selected') else denominator)
    return risk, budget


def select(gain, net, denominator, eligible, mode, *, count=None, seconds=5.):
    risk, budget = coefficients(net, denominator, mode)
    g, ok = np.asarray(gain, float), np.asarray(eligible)
    if (g.shape != risk.shape or ok.shape != risk.shape or ok.dtype != bool
            or not np.isfinite(g).all() or (g[ok] <= 0).any()
            or not np.isfinite(seconds) or seconds <= 0
            or (count is not None and (isinstance(count, bool) or int(count) != count
                                      or not 0 <= count <= int(ok.sum())))):
        raise ValueError('Finite positive eligible gains and valid fixed count required')
    active = np.flatnonzero(ok)

    def exact_result(bits, status, feasible=True):
        actual, allowance = direct_risk(net, denominator, bits, mode)
        return bits, dict(status=status, optimal=feasible, canonical_optimal_verified=feasible,
            solver_reported_optimal=False, failed_closed=not feasible,
            selected=int(bits.sum()), predicted_risk=actual, budget=allowance,
            constraint_pass=actual <= allowance, exact_count_pass=count is None or bool(bits.sum() == count),
            direct_predicted_risk=actual, direct_budget=allowance,
            direct_constraint_pass=actual <= allowance, mode=mode)

    if count == 0 or not len(active):
        return exact_result(np.zeros(len(ok), bool), 'empty_exact_original_inequality')
    whole, allowance = direct_risk(net, denominator, ok, mode)
    if (count is None or count == len(active)) and whole <= allowance:
        return exact_result(ok.copy(), 'all_supported_exact_original_inequality')
    if len(active) <= 10:
        best = None; value = -np.inf
        for v in itertools.product([False, True], repeat=len(active)):
            b = np.zeros(len(ok), bool); b[active] = v
            if count is not None and b.sum() != count: continue
            actual, allowance = direct_risk(net, denominator, b, mode)
            score = math.fsum(g[b])
            if actual <= allowance and score > value: best, value = b, score
        if best is not None: return exact_result(best, 'exhaustive_exact_original_inequality')
        return exact_result(np.zeros(len(ok), bool), 'exact_count_infeasible_floor', False)
    bits, report = allocate(gain, risk, eligible, budget, count=count, seconds=seconds)
    reported_optimal = report['optimal']
    actual, allowance = direct_risk(net, denominator, bits, mode)
    if actual > allowance:
        # The transformed selected-denominator row is not permission to relax
        # its original floating-point inequality by even a rounding tolerance.
        prior = report
        bits[:] = False
        actual, allowance = direct_risk(net, denominator, bits, mode)
        report = dict(status='direct_budget_rounding_floor', optimal=False, selected=0,
            predicted_risk=0., budget=budget, constraint_pass=True,
            exact_count_pass=count is None or count == 0, original=prior)
    # A numerically optimal transformed row does not prove canonical original-
    # inequality optimality; keep a feasible solution without that stronger claim.
    failed = not report['optimal']
    return bits, dict(report, optimal=False, canonical_optimal_verified=False,
        solver_reported_optimal=bool(reported_optimal), failed_closed=failed,
        direct_predicted_risk=actual, direct_budget=allowance,
        direct_constraint_pass=actual <= allowance, mode=mode)
