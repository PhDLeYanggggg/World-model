"""Train on count plans that obey the same token dynamics as inference."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_streaming_menu_capacity import respects_tokens, token_rows


def teacher(gain, risk, easy, legal, offset, easy_cap, mass, seconds=2.):
    q, c = np.nonzero(legal[:, 1:])
    c += 1
    empty = np.zeros(len(legal), dtype=np.int64)
    eligible = legal[:, 1]
    g, r, e = gain[q, c].T, risk[q, c].T, easy[q, c]
    if mass <= 0. or not e.any() or eligible.sum() < 5:
        return empty, "no_support"
    n, nq = len(c), len(legal)
    one = coo_array((np.ones(n), (q, np.arange(n))), shape=(nq, n+1))
    by_q = coo_array((c.astype(float), (q, np.arange(n))), shape=one.shape)
    scale = np.abs(r).max(1).clip(1.)
    main = vstack((one, csc_array(np.column_stack((r/scale[:, None], np.zeros(3)))),
        csc_array(np.r_[e, 0.][None]), csc_array(np.column_stack((-g/mass, np.ones(2))))), format="csc")
    token, upper = token_rows(by_q, eligible)
    mat = vstack((hstack((main, csc_array((main.shape[0], nq)))), token), format="csc")
    low = np.r_[np.full(nq+3, -np.inf), 1., -np.inf, -np.inf, np.full(2*nq, -np.inf)]
    high = np.r_[np.ones(nq), 0., 0., easy_cap/scale[2], np.inf, np.array([0., offset])/mass, upper]
    sol = milp(c=np.r_[np.zeros(n), -1., np.zeros(nq)], integrality=np.r_[np.ones(n), np.zeros(nq+1)],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf, np.zeros(nq)], np.r_[np.ones(n), np.inf, np.full(nq, 10.)]),
        constraints=LinearConstraint(mat, low, high), options=dict(time_limit=seconds, node_limit=400, mip_rel_gap=0.))
    if sol.x is not None:
        at = sol.x[:n] > .5
        counts = np.bincount(q[at], weights=c[at], minlength=nq).astype(np.int64)
        harm = r[:, at].sum(1)
        if (len(np.unique(q[at])) == at.sum() and e[at].any() and respects_tokens(counts, eligible)
                and (harm[:2] <= 0.).all() and harm[2] <= easy_cap
                and 100.*(g[:, at].sum(1)+[0., offset]).min()/mass > 1e-6):
            return counts, "verified_witness"
    if sol.status == 2:
        return empty, "infeasible"
    dual = getattr(sol, "mip_dual_bound", None)
    if dual is not None and -100.*dual <= 1e-6:
        return empty, "nonpositive_bound"
    return empty, "unresolved"


def online(logits, legal, rec):
    """Each row may score all possible current credits; choose only the actual state."""
    state, counts = np.empty(len(rec), dtype=int), np.zeros(len(rec), dtype=int)
    bal = 0
    for i, name in enumerate(rec):
        if i == 0 or name != rec[i-1]:
            bal = 0
        bal = min(10, bal+int(legal[i, 1]))
        state[i] = bal
        on = legal[i] & (np.arange(3)*5 <= bal)
        counts[i] = np.where(on, logits[i, bal], -np.inf).argmax()
        bal -= counts[i]*5
    return counts, state


def teacher_features(plan, legal, rec):
    """Return credit state along the target plan; reject impossible labels."""
    goal = np.maximum(plan, 0)
    one = np.full((len(goal), 3), -2.)
    one[:, 0] = -1.
    one[np.arange(len(goal)), goal] = 1.
    y, state = online(np.broadcast_to(one[:, None], (len(goal), 11, 3)), legal, rec)
    return y, state
