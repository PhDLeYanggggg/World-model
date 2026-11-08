"""Causal two-query deferral with recording-level counterfactual training."""

import numpy as np

from src.world_model.m3w_release_wait import AGE


def walk(query, rows, slots, proxy, release, anchor, eligible, cap, decide):
    """Return chosen edges and release-state features, with a reserve for untouched edits."""
    order = np.argsort(query, kind="stable")
    off = np.r_[0, np.bincount(query, minlength=len(cap)).cumsum()]
    rpos = np.r_[0, np.bincount(release, minlength=len(cap)).cumsum()]
    status = np.zeros(len(release), dtype=bool)
    chosen = []
    pending = np.empty(0, dtype=int)
    deferred = np.zeros(len(release), dtype=bool)
    state = np.zeros((len(release), 5), dtype=np.float32)
    ref = balance = 0
    for t in range(len(cap)):
        arrived = np.arange(rpos[t], rpos[t+1])
        ref = min(10, ref+int(eligible[t]))-5*len(arrived)
        balance = min(10, balance+int(eligible[t]))
        pending = pending[~status[pending] & (release[pending]+AGE >= t)]
        if len(arrived):
            state[arrived] = (balance/10., ref/10., len(pending)/4., cap[t]/2., len(arrived)/2.)
            deferred[arrived] = decide(arrived, state[arrived])
        keep = anchor[arrived[~deferred[arrived]]]
        pending = np.r_[pending, arrived[deferred[arrived]]]
        balance -= 5*len(keep)
        if not 0 <= ref <= balance:
            raise ValueError("An untouched edit cannot be funded")
        chosen.extend(keep)
        status[slots[keep]] = True
        used = set(rows[keep])
        limit = min((balance-ref)//5, cap[t]-len(keep))
        if not limit or not len(pending):
            continue
        ix = order[off[t]:off[t+1]]
        rs = slots[ix]
        on = ix[(proxy[ix] > 0.) & deferred[rs] & ~status[rs] &
                (release[rs] < t) & (release[rs]+AGE >= t)]
        for e in on[np.argsort(-proxy[on], kind="stable")]:
            if rows[e] in used or status[slots[e]]:
                continue
            used.add(rows[e])
            status[slots[e]] = True
            chosen.append(e)
            balance -= 5
            limit -= 1
            if not limit:
                break
    return np.asarray(chosen, dtype=int), state, deferred


def local_wait(query, slot, proxy, release, anchor):
    """First positive opportunity, deliberately ignoring competition and credit."""
    on = np.flatnonzero((query > release[slot]) & (proxy > 0.))
    ranked = on[np.lexsort((-proxy[on], query[on], slot[on]))]
    key, i = np.unique(slot[ranked], return_index=True)
    choice = np.full(len(anchor), -1, dtype=int)
    choice[key] = ranked[i]
    return choice
