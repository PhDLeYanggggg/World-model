"""Chronological two-budget allocation with exact current-query decisions."""
import math

import numpy as np

EPS = 1e-10  # Floating-point comparison only; not a change to the risk threshold.


def totals(choice, value, all_excess, easy_excess):
    ix = np.flatnonzero(choice >= 0); take = (ix, choice[ix])
    return np.array([math.fsum(t[take].tolist()) for t in (value, all_excess, easy_excess)])


def pareto(value, a, e, allowed):
    v0, v1 = value[:, :, None], value[:, None, :]
    a0, a1 = a[:, :, None], a[:, None, :]
    e0, e1 = e[:, :, None], e[:, None, :]
    k = np.arange(value.shape[1])
    strict = (v0 > v1) | (a0 < a1) | (e0 < e1) | (k[:, None] < k[None])
    dominated = ((v0 >= v1) & (a0 <= a1) & (e0 <= e1) & strict & allowed[:, :, None]).any(1)
    return allowed & ~dominated


def query_choice(value, a, e, allowed, slack, count=None):
    """Maximize utility for 0/1/2 switches, never selecting two actions for a row."""
    if (value.ndim != 2 or not len(value) or any(x.shape != value.shape for x in (a, e, allowed))
            or allowed.dtype != bool or not np.isfinite([value, a, e]).all()
            or np.shape(slack) != (2,) or not np.isfinite(slack).all() or np.any(slack < -EPS)):
        raise ValueError('Finite causal costs and a two-budget balance required')
    cap = min(2, len(value)-1)
    if count is not None and not 0 <= count <= cap:
        raise ValueError('Matched count must respect the original query cap')
    out = np.full(len(value), -1, dtype=np.int16)
    if count == 0 or cap == 0:
        return out
    rows, acts = np.where(pareto(value, a, e, allowed))
    order = np.lexsort((acts, rows, -value[rows, acts])); r, k = rows[order], acts[order]
    u, x, y = (v[r, k] for v in (value, a, e)); aa, ee = np.asarray(slack)+EPS
    best, take = (0., ()) if count is None else (-np.inf, ())
    if count != 2:
        single = np.flatnonzero((x <= aa) & (y <= ee))
        if len(single) and u[single[0]] > best:
            best, take = u[single[0]], (single[0],)
    if cap == 2 and count != 1:
        # Sort by utility: the first feasible partner is the best for this action.
        for j in range(len(r)-1):
            if u[j]+u[j+1] <= best:
                break
            s = slice(j+1, None)
            partner = (r[s] != r[j]) & (x[s]+x[j] <= aa) & (y[s]+y[j] <= ee)
            if partner.any():
                t = j+1+int(partner.argmax()); gain = u[j]+u[t]
                if gain > best:
                    best, take = gain, (j, t)
    if count is not None and len(take) != count:
        return None
    ix = np.asarray(take, dtype=int); out[r[ix]] = k[ix]
    return out


def stream(groups, value, a, e, allowed, *, counts=None, carry=True, witness=None):
    """Past query slack can fund current actions; a suffix never affects a prefix."""
    ledger = np.zeros(2); out = np.full(len(value), -1, dtype=np.int16)
    fallbacks = 0; max_local = np.zeros(2); credit_spent = 0; gain = 0.
    for k, g in enumerate(groups):
        slack = ledger.copy() if carry else np.zeros(2)
        u, ac, ec, legal = (z[g] for z in (value, a, e, allowed))
        count = None if counts is None else int(counts[k])
        choice = query_choice(u, ac, ec, legal, slack, count)
        if choice is None:
            if witness is None:
                raise ValueError('No matched prediction-budget witness')
            choice = witness[g].copy(); fallbacks += 1
        used = np.flatnonzero(choice >= 0)
        if not legal[used, choice[used]].all() or count is not None and len(used) != count:
            raise ValueError('Invalid candidate or matched count')
        g0, *cost = totals(choice, u, ac, ec); cost = np.asarray(cost)
        if (cost > slack+EPS).any():
            raise ValueError('Predicted budget prefix violated')
        if (cost > EPS).any():
            credit_spent += 1
        ledger -= cost
        max_local = np.maximum(max_local, cost)
        gain += g0; out[g] = choice
    return out, dict(predicted_utility=float(gain), predicted_final_slack=ledger.tolist(),
                     max_positive_query_excess=max_local.tolist(), queries_using_credit=credit_spent,
                     witness_fallback_queries=fallbacks)
