"""Hindsight feasibility of acquiring one easy reference and pruning direct actions."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, hstack, vstack

from src.evaluation.m3w_prefix_capacity import incidence
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments

MODES = ('insert_only', 'early_joint', 'ordered_joint', 'free_joint', 'shortlist_joint')


def pool(floor, bank, target, valid, cut, choices, groups, shortlist):
    n, k = bank.shape[:2]
    at = np.empty(n, dtype=int)
    for j, q in enumerate(groups):
        at[q] = j
    size = np.asarray([len(q) for q in groups])
    risk_keys = ('all_excess', 'easy_excess', 'easy_signed')
    m = [moments(floor, bank[:, f], target, valid, cut) for f in range(k)]
    easy = m[0]['ref_easy'] > 0
    selected = np.flatnonzero(choices >= 0)
    if easy[selected].any():
        raise ValueError('Capacity diagnosis requires missing selected easy support')
    chosen_family = choices[selected]
    d = {key: np.column_stack([z[key] for z in m])[selected, chosen_family]
         for key in (*risk_keys, 'known', 'diff', 'radius')}
    r, a = np.where(easy[:, None] & (choices[:, None] < 0) &
                    (np.column_stack([z['radius'] for z in m]) > 0) &
                    (np.column_stack([z['easy_excess'] for z in m]) <= 0.))
    cost = np.stack([np.column_stack([z[key] for z in m])[r, a] for key in risk_keys])
    gain = np.column_stack([z['diff'] for z in m])[r, a] / size[at[r]]
    return dict(selected=selected, selected_query=at[selected],
                selected_cost=np.stack([d[k] for k in risk_keys]),
                deletion_gain=np.where(d['known'], -d['diff'], -d['radius'])/size[at[selected]],
                row=r, action=a, query=at[r], cost=cost, gain=gain, shortlist=shortlist[r, a],
                cap=np.minimum(2, size-1), full_easy=float(m[0]['ref_easy'].sum()))


def pareto(query, excess, gain, allowed):
    """Same-query dominance is exact with one inserted easy action."""
    idx = np.flatnonzero(allowed)
    order = idx[np.lexsort((-gain[idx], excess[idx], query[idx]))]
    take, last, high = [], -1, -np.inf
    for i in order:
        if query[i] != last:
            last, high = query[i], -np.inf
        if gain[i] > high:
            take.append(i)
            high = gain[i]
    return np.asarray(take, dtype=int)


def solve(problem, mode, seconds=4.):
    if mode not in MODES:
        raise ValueError('Unknown comparison')
    p = problem
    dq = p['selected_query']
    nq = len(p['cap'])
    allowed = np.ones(len(p['query']), dtype=bool)
    if mode == 'early_joint' and len(dq):
        allowed &= p['query'] < dq.min()
    if mode == 'shortlist_joint':
        allowed &= p['shortlist']
    ids = pareto(p['query'], p['cost'][1], p['gain'], allowed)
    if not len(ids):
        return None, 'no_eligible_anchor', 0
    aq = p['query'][ids]
    n, k = len(dq), len(ids)
    before = np.bincount(dq, minlength=nq)
    query = hstack((-incidence(dq, nq), incidence(aq, nq)), format='csc')
    risk = np.column_stack((-p['selected_cost'], p['cost'][:, ids]))
    gain = np.r_[p['deletion_gain'], p['gain'][ids]]
    units = np.maximum(np.abs(risk).max(1), 1.)
    upper_risk = np.array([0., 0., BUDGET*p['full_easy']])-p['selected_cost'].sum(1)
    blocks = [query, incidence(dq, nq), csc_array(risk/units[:, None]),
              csc_array(-gain[None]/max(1., np.abs(gain).max())),
              csc_array(np.r_[np.zeros(n), np.ones(k)][None])]
    blocks[1] = hstack((blocks[1], csc_array((nq, k))), format='csc')
    lo = np.r_[[-np.inf]*(2*nq+4), 1.]
    hi = np.r_[p['cap']-before, np.ones(nq), upper_risk/units, 0., 1.]
    if mode in ('early_joint', 'ordered_joint', 'shortlist_joint') and n:
        # A removal is allowed only once the inserted support has arrived.
        blocks.append(hstack((incidence(np.arange(n), n),
                              csc_array(-(aq[None] <= dq[:, None]).astype(float))), format='csc'))
        lo = np.r_[lo, [-np.inf]*n]
        hi = np.r_[hi, np.zeros(n)]
    upper = np.r_[np.full(n, 0. if mode == 'insert_only' else 1.), np.ones(k)]
    res = milp(c=np.r_[np.ones(n), np.zeros(k)], integrality=np.ones(n+k),
               bounds=Bounds(0., upper), constraints=LinearConstraint(vstack(blocks, format='csc'), lo, hi),
               options=dict(time_limit=seconds, mip_rel_gap=0.))
    if res.x is None:
        return None, 'infeasible' if res.status == 2 else 'unresolved', k
    pick = np.flatnonzero(res.x[n:] > .5)
    if len(pick) != 1:
        return None, 'unresolved_postcheck', k
    return (res.x[:n] > .5, ids[pick[0]]), 'optimal' if res.status == 0 else 'feasible', k
