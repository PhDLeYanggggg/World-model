"""Causal allocation of an exogenous stream of per-query edit credits."""

from collections import deque

import numpy as np


POLICIES = ("immediate", "fixed_wait", "value_wait", "support_wait")


def legal_schedule(release, emitted, capacity, delay, complete):
    release, emitted, capacity = map(np.asarray, (release, emitted, capacity))
    cum = release.cumsum()
    spend = emitted.cumsum()
    due = np.r_[np.zeros(delay, dtype=int), cum][:len(release)]
    return bool((emitted >= 0).all() and (emitted <= capacity).all() and (spend <= cum).all()
                and (not complete or ((spend >= due).all() and spend[-1] == cum[-1])))


def stream(packets, policy, delay=2):
    """Each packet exposes only the current legal rows, scores and released count."""
    if policy not in POLICIES or delay < 1:
        raise ValueError("Invalid registered credit policy")
    waiting = deque()
    events = []
    expired = 0
    no_support = 1.
    for q, (rows, count, value, support) in enumerate(packets):
        if (len(rows) != len(value) or len(rows) != len(support) or not 0 <= count <= len(rows)
                or not np.isfinite(value).all() or not np.isfinite(support).all()
                or (support < 0.).any() or (support > 1.).any()):
            raise ValueError("Unaligned current-query release packet")
        order = np.argsort(-value, kind="stable")
        waiting.extend((q, int(row)) for row in rows[order[:count]])
        capacity = count if policy == "immediate" else min(count+1, len(rows))
        for j in order[:capacity]:
            if not waiting:
                break
            start, original = waiting[0]
            age = q-start
            if policy == "fixed_wait" and age < 1:
                break
            if policy in ("value_wait", "support_wait") and age < delay:
                if value[j] <= 0. or (policy == "support_wait" and no_support >= .5 and support[j] < .5):
                    break
            waiting.popleft()
            events.append((int(rows[j]), original, q, age))
            no_support *= 1.-support[j]
        while waiting and q-waiting[0][0] >= delay:
            waiting.popleft()
            expired += 1
    return events, expired+len(waiting)
