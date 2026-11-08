"""Online support-gated release and finite matching diagnostics."""

import numpy as np

from src.world_model.m3w_opendd_two_reserves import EASY_THRESHOLD
from src.world_model.m3w_opendd_record_support import support_targets


def earliest_gain(packets, excluded=None):
    for rows, gain, _, _ in packets:
        take = (rows != excluded) & (gain > 0.)
        if take.any():
            return int(rows[np.where(take, gain, -np.inf).argmax()])
    return None


def stream(packets, learned=False):
    """Packets are current rows, gain, easy probability and local-support probability."""
    support = None
    for rows, gain, easy, prob in packets:
        if support is None:
            use = (easy >= EASY_THRESHOLD) & ((prob >= .5) if learned else True)
            if not use.any():
                continue
            rank = prob if learned else easy
            support = int(rows[np.where(use, rank, -np.inf).argmax()])
        use = (rows != support) & (gain > 0.)
        if use.any():
            return int(rows[np.where(use, gain, -np.inf).argmax()]), support
    return None, support


def witness(big, small, query, legal, budget, before_gain, ref_easy, ols_offset):
    """Exact one/two-role, outcome-aware witness; never used by stream."""
    n_big, n_small = budget
    if n_big not in (0, 1) or n_small not in (0, 1):
        raise ValueError("Only one edit per role is supported")
    ids = np.flatnonzero(legal)
    if ref_easy <= 0. or not (n_big or n_small) or not len(ids):
        return None
    if not (n_big and n_small):
        bank = big if n_big else small
        _, ok = support_targets(*(x[ids] for x in bank), ref_easy, ols_offset=ols_offset)
        if not ok.any():
            return None
        chosen = int(ids[np.flatnonzero(ok)[0]])
        return (chosen, None) if n_big else (None, chosen)
    for j in ids:
        pool = ids[(ids != j) & ((query[ids] <= query[j]) if before_gain else True)]
        if not len(pool):
            continue
        _, ok = support_targets(*(x[pool] for x in small), ref_easy,
                                first=tuple(x[j] for x in big), ols_offset=ols_offset)
        if ok.any():
            return int(j), int(pool[np.flatnonzero(ok)[0]])
    return None
