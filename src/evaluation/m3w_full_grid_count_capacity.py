"""Exact fixed-spend feasibility with original and complete-grid risk."""

from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_count_motion_capacity import within_cap
from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP, respects_tokens, token_rows


def enumerate_spend(legal, movement, total, cap, *, max_plans=25000):
    """Enumerate small cases without solver tolerances beyond the archived cap."""
    active = np.flatnonzero(legal[:, 1])
    sizes = range((total+1)//2, min(total, len(active))+1)
    if sum(comb(len(active), k)*comb(k, total-k) for k in sizes) > max_plans:
        return None

    def plans():
        rows = np.arange(len(legal))
        for k in sizes:
            for nonzero in combinations(active, k):
                two = tuple(i for i in nonzero if legal[i, 2])
                for double in combinations(two, total-k):
                    n = np.zeros(len(legal), dtype=int)
                    n[list(nonzero)] = 1
                    n[list(double)] = 2
                    if within_cap(movement[rows, n].sum(), cap) and respects_tokens(n, legal[:, 1]):
                        yield n

    return plans()


def prefix_grid_terms(cv, path, target, valid, rows, cutoff, limit=.02):
    """Each chosen option adds two worst-case excesses and mandatory support."""
    use = rows >= 0
    ix = rows[use]
    if (rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(cutoff) or cutoff <= 0.
            or len(ix) != len(set(ix)) or (ix >= len(cv)).any() or (use[:, 1] & ~use[:, 0]).any()):
        raise ValueError('Expected distinct frozen prefix nominees and a positive cutoff')
    c, p, y, v = (z[ix] for z in (cv, path, target, valid))
    y = np.where(v[..., None], y, c)
    distance = np.linalg.norm(p-c, axis=-1)
    if (distance.max(1) <= 0.).any():
        raise ValueError('Frozen nominees must be nonidentity edits')
    base = np.where(v, np.linalg.norm(c-y, axis=-1), 0.)
    change = np.where(v, np.linalg.norm(p-y, axis=-1)-base, distance)
    floor = base.sum(1)
    excess = np.maximum(change.sum(1), 0.)-limit*floor
    complete = v.all(1)
    in_range = floor/12 <= cutoff
    mandatory = complete & (floor > 0.) & in_range
    optional = ~complete & in_range
    easy = np.where(mandatory, excess, np.where(optional, np.maximum(excess, 0.), 0.))
    terms = np.zeros((*rows.shape, 3))
    terms[use] = np.column_stack((excess, easy, mandatory))
    return np.concatenate((np.zeros_like(terms[:, :1]), terms.cumsum(1)), axis=1)


def solve(legal, movement, total, cap, terms, grid, *, seconds=2.):
    """Maximize the lower gain against both references; never return unchecked exclusions."""
    if not total:
        return None, 'zero_realized'
    gain, original_risk, _, offset, easy_cap, mass = terms
    q, n = np.nonzero(legal[:, 1:] & within_cap(movement[:, 1:], cap))
    n += 1
    if not len(n) or mass <= 0. or not grid[q, n, 2].any():
        return None, 'no_mandatory_easy_support'
    nq, k = len(legal), len(n)
    cols = np.arange(k)
    choice = coo_array((np.ones(k), (q, cols)), shape=(nq, k+1))
    spend = coo_array((n.astype(float), (q, cols)), shape=choice.shape)
    r = np.column_stack((original_risk[q, n], grid[q, n, :2])).T
    scale = np.abs(r).max(1).clip(1.)
    g = gain[q, n].T
    blocks = (choice, csc_array(np.r_[n, 0.][None]), csc_array(np.r_[movement[q, n]/cap, 0.][None]),
              csc_array(np.column_stack((r/scale[:, None], np.zeros(5)))),
              csc_array(np.r_[grid[q, n, 2], 0.][None]),
              csc_array(np.column_stack((-g/mass, np.ones(2)))))
    lower = np.r_[np.full(nq, -np.inf), total, np.full(6, -np.inf), 1., -np.inf, -np.inf]
    upper = np.r_[np.ones(nq), total, 1.+1e-9, 0., 0., easy_cap/scale[2], 0., 0., np.inf, offset/mass]
    core = vstack(blocks, format='csc')
    tail, available = token_rows(spend, legal[:, 1])
    matrix = vstack((hstack((core, csc_array((core.shape[0], nq)))), tail), format='csc')
    ans = milp(c=np.r_[np.zeros(k), -1., np.zeros(nq)], integrality=np.r_[np.ones(k), np.zeros(1+nq)],
        bounds=Bounds(np.r_[np.zeros(k), -np.inf, np.zeros(nq)], np.r_[np.ones(k), np.inf, np.full(nq, 10.)]),
        constraints=LinearConstraint(matrix, np.r_[lower, np.full(2*nq, -np.inf)], np.r_[upper, available]),
        options=dict(time_limit=seconds, mip_rel_gap=0., presolve=False))
    if ans.x is not None:
        active = ans.x[:k] > .5
        count = np.zeros(nq, dtype=int)
        count[q[active]] = n[active]
        if (len(set(q[active])) != active.sum() or count.sum() != total
                or not respects_tokens(count, legal[:, 1]) or not within_cap(movement[q[active], n[active]].sum(), cap)):
            return None, 'unresolved_numerical_check'
        if float(100.*np.min(offset+g[:, active].sum(1))/mass) > POSITIVE_PP:
            return count, 'candidate'
    if ans.status == 2:
        return None, 'infeasible_full_grid'
    bound = getattr(ans, 'mip_dual_bound', None)
    if bound is not None and np.isfinite(bound) and -100.*bound <= POSITIVE_PP:
        return None, 'no_strict_positive_bound'
    return None, 'unresolved'
