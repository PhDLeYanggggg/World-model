"""Exact path-pair search on a fixed recipient shortlist."""
import numpy as np

from src.world_model.m3w_record_joint_actions import PAIR_WEIGHT, local_edges


def options(history, cv, ols, bank, scale, u, a, e, support, legal, k):
    """Return all exact-count assignments and past-only scores."""
    if k not in (1, 2):
        raise ValueError('Only one or two switches may be enumerated')
    row, path = np.nonzero(legal)
    if k == 1:
        combos = np.arange(len(row))[:, None]
    else:
        p, q = np.triu_indices(len(row), 1)
        use = row[p] != row[q]
        combos = np.column_stack((p[use], q[use]))
    agent, candidate = row[combos], path[combos]
    i, j = local_edges(history)
    degree = np.bincount(np.r_[i, j], minlength=len(cv))
    weight = 1./np.sqrt(degree[i]*degree[j])
    sd = (scale[i]*scale[j])[:, None, None]
    delta = bank-cv[:, None]
    discrepancy = (cv[i]-cv[j])-(ols[i]-ols[j])
    left, right = delta[i], delta[j]
    marginal = np.zeros_like(u)
    np.add.at(marginal, i, weight[:, None]*((2.*discrepancy[:, None]*left+left**2)/sd[:, None]).sum(-1).mean(-1))
    np.add.at(marginal, j, weight[:, None]*((-2.*discrepancy[:, None]*right+right**2)/sd[:, None]).sum(-1).mean(-1))
    adjusted = u/scale[:, None] - PAIR_WEIGHT*marginal
    additive = adjusted[agent, candidate].sum(1)
    joint = additive.copy()
    if k == 2:
        edge = np.zeros((len(cv), len(cv)))
        edge[i, j] = weight
        l, r = agent.T
        dot = (delta[agent[:, 0], candidate[:, 0]]*delta[agent[:, 1], candidate[:, 1]]).sum(-1).mean(1)
        joint += 2.*PAIR_WEIGHT*edge[l, r]*dot/(scale[l]*scale[r])
    admit = ((u[agent, candidate].sum(1) > 0.) & (a[agent, candidate].sum(1) <= 0.)
             & (e[agent, candidate].sum(1) <= 0.) & (support[agent, candidate].sum(1) > 0.))
    return agent, candidate, additive, joint, admit


def choose(history, cv, ols, bank, scale, u, a, e, support, legal, k, old, active):
    """Compare gated alternatives at the old activation mask and at free activation."""
    raw, additive, joint, late = [np.full(len(cv), -1, dtype=np.int16) for _ in range(4)]
    if not k:
        return raw, additive, joint, late, 0, 0
    who, how, va, vj, admit = options(history, cv, ols, bank, scale, u, a, e, support, legal, k)
    p = vj.argmax()
    raw[who[p]] = how[p]
    at = np.flatnonzero(old >= 0)
    old_option = (who == at[None]).all(1) & (how == old[at][None]).all(1)
    if old_option.sum() != 1 or vj[p]+1e-10 < vj[old_option][0]:
        raise ValueError('The old decoded option must be contained in the exact search')
    if bool(admit[old_option][0]) != bool(active):
        raise ValueError('Frozen gate does not match on the old option')
    allowed = np.flatnonzero(admit)
    if len(allowed):
        j = allowed[vj[allowed].argmax()]
        late[who[j]] = how[j]
        if active:
            joint[:] = late
            i = allowed[va[allowed].argmax()]
            additive[who[i]] = how[i]
    return raw, additive, joint, late, len(who), len(allowed)
