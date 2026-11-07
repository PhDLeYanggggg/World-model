"""Separate ranking and agent-diversity effects on shortlist retention."""
import numpy as np


def mix(family, portfolio):
    q, i, j, x = family
    q2, u, v, y = portfolio
    if not np.array_equal(q, q2) or len(i) != len(u):
        raise ValueError('Matched query and shortlist size required')
    if i[0] != u[0] or j[0] != v[0]:
        raise ValueError('Shortlists must share their first choice')
    chosen = {}
    for rows, fam, feat in ((i[:6], j[:6], x[:6]), (u, v, y), (i, j, x)):
        for row, k, desc in zip(rows, fam, feat):
            if len(chosen) == len(i):
                break
            chosen.setdefault((row, k), desc)
    a, b = np.asarray(list(chosen)).T
    z = np.stack(list(chosen.values()))
    z[:, 9] = np.arange(len(z))/8
    return q, a, b, z


def alternatives(joint, gain, all_excess, easy_excess, legal):
    """One-factor counterfactuals; neither reads labels nor changes a budget."""
    ok = np.flatnonzero(legal.ravel())
    n = legal.shape[1]
    j, u = joint.ravel()[ok], gain.ravel()[ok]
    excess = (np.maximum(all_excess, 0.) + np.maximum(easy_excess, 0.)).ravel()[ok]
    flat = ok[np.argsort(-j, kind='stable')]
    first, rest, families = [], [], set()
    for i in flat:
        family = i % n
        (rest if family in families else first).append(i)
        families.add(family)
    first_by_row, other, seen = [], [], set()
    for i in first + rest:
        agent = i // n
        (other if agent in seen else first_by_row).append(i)
        seen.add(agent)
    orders = [ok[np.lexsort((-j, -v))] for v in (j, u, u-excess, -excess)]
    selected = {}
    for k in range(min(8, len(ok))):
        take = next(i for i in orders[k % len(orders)] if i not in selected)
        selected[take] = None
    return np.asarray((first_by_row+other)[:8], dtype=int), np.asarray(list(selected), dtype=int)


def mechanism(family_with_agent, cost_without_agent):
    return {
        (True, True): 'combination_only',
        (True, False): 'cost_alone_sufficient',
        (False, True): 'agent_alone_sufficient',
        (False, False): 'either_alone_sufficient',
    }[bool(family_with_agent), bool(cost_without_agent)]
