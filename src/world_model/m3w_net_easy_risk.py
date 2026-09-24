"""Signed easy-risk accounting, not a finite-sample safety certificate."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.evaluation.m3w_easy_moment import targets as positive_targets


def targets(costs, cv, distance, known, cutoff):
    old = positive_targets(costs, cv, distance, known, cutoff)
    d = np.asarray(distance)
    easy = np.asarray(known) & (np.asarray(cv) <= cutoff)
    benefit = np.divide(np.where(easy, np.asarray(costs)[:, 0], 0), d,
                        out=np.zeros(len(d)), where=d > 0).clip(0, 1)
    return np.column_stack((old[:, 0], benefit, old[:, 1], old[:, 2]))


def moments(fractions, distance, cutoff):
    f, d = np.asarray(fractions), np.asarray(distance)
    if (f.shape != (len(d), 4) or d.ndim != 1 or not np.isfinite(f).all()
            or not np.isfinite(d).all() or (d < 0).any() or (f < 0).any()
            or (f > 1 + 1e-10).any() or (f[:, 0] + f[:, 1] > f[:, 3] + 1e-10).any()
            or (f[:, 2] > f[:, 3] + 1e-10).any() or not np.isfinite(cutoff) or cutoff <= 0):
        raise ValueError('Coherent causal easy moments required')
    return f[:, 0] * d, (f[:, 0] - f[:, 1]) * d, f[:, 2] * cutoff


def allocate(gain, risk, eligible, budget, *, count=None, seconds=5.):
    """Maximize source-predicted net gain under one query-local signed risk row.

    Negative risk can fund another intervention in this same query, never in a
    different recording/frame. No pruning by individual positive risk is valid.
    """
    g, q, ok = np.asarray(gain, float), np.asarray(risk, float), np.asarray(eligible)
    if (g.ndim != 1 or q.shape != g.shape or ok.shape != g.shape or ok.dtype != bool
            or not np.isfinite(g).all() or not np.isfinite(q).all()
            or not np.isfinite(budget) or budget < 0 or (g[ok] <= 0).any()
            or not np.isfinite(seconds) or seconds <= 0
            or (count is not None and (isinstance(count, bool) or int(count) != count
                                      or not 0 <= count <= int(ok.sum())))):
        raise ValueError('Finite source-only supported gains and nonnegative query allowance required')
    rows = np.flatnonzero(ok)
    bits = np.zeros(len(g), bool)
    tol = 1e-9 * max(abs(budget), float(np.abs(q[rows]).sum()), 1e-12)

    def result(status, optimal, numeric=None):
        return bits.copy(), dict(status=status, optimal=optimal, selected=int(bits.sum()),
            predicted_risk=float(q[bits].sum()), budget=float(budget),
            constraint_pass=bool(q[bits].sum() <= budget + tol),
            exact_count_pass=count is None or int(bits.sum()) == count, numerical=numeric)

    if not len(rows) or count == 0:
        return result('empty_exact', True)
    if (count is None or count == len(rows)) and q[rows].sum() <= budget:
        bits[rows] = True
        return result('all_supported_exact', True)
    # Normalize risk independently of objective to avoid HiGHS absolute-unit
    # tolerances accepting a material overrun on tiny annotation-pixel budgets.
    rs = max(float(np.abs(q[rows]).max()), budget, 1e-12)
    gs = max(float(g[rows].max()), 1e-12)
    matrix = [q[rows] / rs]; lower = [-np.inf]; upper = [budget / rs]
    if count is not None:
        matrix.append(np.ones(len(rows))); lower.append(count); upper.append(count)
    solved = milp(-g[rows] / gs, integrality=np.ones(len(rows)),
        bounds=Bounds(np.zeros(len(rows)), np.ones(len(rows))),
        constraints=LinearConstraint(np.array(matrix), np.array(lower), np.array(upper)),
        options=dict(time_limit=float(seconds), mip_rel_gap=0.))
    if solved.status != 0 or solved.x is None:
        return result('solver_nonoptimal_floor', False)
    rounded = np.rint(solved.x)
    if not np.isfinite(solved.x).all() or np.max(np.abs(solved.x-rounded)) > 1e-7:
        return result('nonintegral_floor', False)
    if not np.isin(rounded, [0, 1]).all():
        return result('invalid_bits_floor', False)
    bits[rows] = rounded.astype(bool)
    primal = -float(g[bits].sum()) / gs
    dual = getattr(solved, 'mip_dual_bound', np.nan)
    objective_tol = 1e-8 * (1 + abs(primal))
    numeric = dict(primal=primal, reported=float(solved.fun), dual=float(dual),
                   gain_scale=gs, risk_scale=rs)
    good = (np.isfinite([primal, solved.fun, dual]).all()
        and abs(primal-solved.fun) <= objective_tol
        and -objective_tol <= primal-dual <= objective_tol
        and q[bits].sum() <= budget+tol
        and (count is None or bits.sum() == count))
    if not good:
        bits[:] = False
        return result('original_unit_check_floor', False, numeric)
    return result('milp_exact', True, numeric)
