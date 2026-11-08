"""Exact-radius alternatives and past-only scheduling of released edit slots."""

import numpy as np

from src.world_model import m3w_radius_matched_turn_menu as shape

WINDOWS = {"fixed": (0, 0), "delay2": (0, 2), "delay8": (0, 8), "move8": (-8, 8)}


def candidates(query, native_peak, cap, release, radius, before, after):
    radius = np.asarray(radius)
    offsets = np.r_[0, np.bincount(query, minlength=len(cap)).cumsum()]
    slots, rows = [], []
    for i, (q, r) in enumerate(zip(release, radius, strict=True)):
        for t in range(max(0, q+before), min(len(cap), q+after+1)):
            if cap[t]:
                ix = np.arange(offsets[t], offsets[t+1])
                ix = ix[(native_peak[ix] > 0.) & (native_peak[ix]+1e-10 >= r)]
                slots.extend([i]*len(ix))
                rows.extend(ix)
    rows, slots = np.array(rows, dtype=int), np.array(slots, dtype=int)
    if not np.array_equal(np.unique(slots), np.arange(len(release))):
        raise ValueError("Original fixed edit must remain available")
    return rows, slots, (radius[slots]/native_peak[rows]).clip(max=1.)


def proposals(history, amp):
    cv, _, bank = shape.make_menu(history, 12, amp)
    proxy = shape.prefix_rank(history, amp, amp > 0.).mean(0)
    k = proxy.argmax(1)
    return cv, bank[np.arange(len(k)), k], proxy[np.arange(len(k)), k]


def schedule(edge_query, rows, slots, score, release, eligible, cap, delay):
    """Spend only earned tokens and use only candidates observable at the current query."""
    if delay < 0:
        raise ValueError("A causal scheduler cannot advance releases")
    nq = len(cap)
    # Keep input order for stable ties; future query scores never enter a decision.
    order = np.argsort(edge_query, kind="stable")
    offsets = np.r_[0, np.bincount(edge_query, minlength=nq).cumsum()]
    spent = np.zeros(len(release), dtype=bool)
    chosen = []
    balance = 0
    for q in range(nq):
        balance = min(10, balance+int(eligible[q]))
        limit = min(balance//5, cap[q])
        if not limit:
            continue
        ix = order[offsets[q]:offsets[q+1]]
        active = ix[(score[ix] > 0.) & (release[slots[ix]] <= q) &
                    (release[slots[ix]]+delay >= q) & ~spent[slots[ix]]]
        used = set()
        for e in active[np.argsort(-score[active], kind="stable")]:
            if rows[e] in used or spent[slots[e]]:
                continue
            chosen.append(e)
            spent[slots[e]] = True
            used.add(rows[e])
            balance -= 5
            if len(used) == limit:
                break
    return np.asarray(chosen, dtype=int)
