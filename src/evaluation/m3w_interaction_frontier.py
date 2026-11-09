"""Exact fixed-budget interaction bounds with recording-level risk constraints."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_exact_slot_headroom import POSITIVE_PP
from src.world_model.m3w_slate_shared_menu import legal_assignments

UTILITY_TOL_PP = 1e-8
INTERACTION_TOL = 1e-10


def pair_reduction(paths, control, target, valid):
    """Known-query pair gain and full-grid triangle-inequality lower gain."""
    if len(control) < 2:
        return np.zeros((len(paths), 2))
    i, j = np.triu_indices(len(control), 1)
    both = valid[i] & valid[j]
    ref = control[i] - control[j]
    rel = paths[:, i] - paths[:, j]
    truth = np.where(both[..., None], target[i] - target[j], 0.)
    delta = np.linalg.norm(ref - truth, axis=-1) - np.linalg.norm(rel - truth, axis=-1)
    known = np.where(both, delta, 0.).sum((1, 2))
    debit = np.where(both, 0., np.linalg.norm(rel - ref, axis=-1)).sum((1, 2))
    return np.column_stack((known / max(1, both.sum()), (known - debit) / both.size))


def denominators(q, valid):
    known = full = 0
    for j in np.unique(q):
        flags = valid[q == j]
        if len(flags) >= 2:
            full += 1
            known += bool((flags.sum(0) >= 2).any())
    return np.array([known, full])


def aggregate_pair(path, control, target, valid, q, den):
    total = sum((pair_reduction(path[None, q == j], control[q == j], target[q == j], valid[q == j])[0]
                 for j in np.unique(q)), start=np.zeros(2))
    return np.divide(total, den, out=np.zeros(2), where=den > 0)


def query_options(design, cv, control, target, valid, q, den):
    """Enumerate complete local choices; one binary variable selects each query's slate."""
    d = design
    action_query = q[d["row"]]
    groups, choices, gains, risks, easy, pair = [], [], [], [], [], []
    for k, j in enumerate(np.unique(action_query)):
        at = np.flatnonzero(q == j)
        slots = [np.flatnonzero(d["enabled"] & (d["slot"] == s))
                 for s in np.unique(d["slot"][action_query == j])]
        flat = np.concatenate(slots)
        if len(slots) == 1:
            # The old double-only comparison already freezes singleton actions.
            if len(flat) != 1:
                raise ValueError("Expected a frozen singleton action")
            take = flat[:, None]
        elif len(slots) == 2:
            order = legal_assignments(d["row"][slots[0]], d["row"][slots[1]])
            take = flat[order]
        else:
            raise ValueError("Require one or two slots per query")
        if not len(take):
            raise ValueError("Frozen query lost its legal choice")
        local = np.searchsorted(at, d["row"][take])
        parts = []
        for first in range(0, len(take), 128):
            selected = take[first:first+128]
            batch = np.broadcast_to(cv[at], (len(selected), *cv[at].shape)).copy()
            batch[np.arange(len(selected))[:, None], local[first:first+128]] = d["path"][selected]
            parts.append(pair_reduction(batch, control[at], target[at], valid[at]))
        p = np.concatenate(parts)
        pair.append(np.divide(p, den, out=np.zeros_like(p), where=den > 0))
        gains.append(d["gain"][take].sum(1))
        risks.append(d["risk"][take].sum(1))
        easy.append(d["easy"][take].sum(1))
        groups.extend([k] * len(take))
        choices.extend(take)
    return dict(group=np.array(groups), actions=choices, gain=np.concatenate(gains),
                risk=np.concatenate(risks), easy=np.concatenate(easy), pair=np.concatenate(pair))


def relaxed_pair_bound(menu):
    groups, first = np.unique(menu["group"], return_index=True)
    if len(groups) == 0 or np.any(np.diff(menu["group"]) < 0):
        raise ValueError("Expected sorted nonempty query groups")
    return np.maximum.reduceat(menu["pair"], first, axis=0).sum(0)


def solve(menu, design, objective, seconds):
    groups = menu["group"]
    n = len(groups)
    query = coo_array((np.ones(n), (groups, np.arange(n))), shape=(groups.max()+1, n))
    scale = np.abs(menu["risk"]).max(0).clip(1e-6)
    extra = csc_array(np.vstack((menu["risk"].T / scale[:, None], -menu["easy"],
                                 -menu["gain"].T)))
    constraints = LinearConstraint(
        vstack((query, extra), format="csc"),
        np.r_[np.ones(query.shape[0]), np.full(7, -np.inf)],
        np.r_[np.ones(query.shape[0]), 0., 0., design["easy_cap"]/scale[2], -1.,
              design["offset"] - np.array([POSITIVE_PP, POSITIVE_PP, -UTILITY_TOL_PP])],
    )
    obj = menu["pair"][:, objective]
    norm = max(np.abs(obj).max(), 1e-9)
    result = milp(-obj/norm, integrality=np.ones(n), bounds=Bounds(0., 1.),
                  constraints=constraints, options={"time_limit": seconds, "mip_rel_gap": 0.})
    status = {0: "optimal", 1: "limited", 2: "infeasible"}.get(result.status, "unresolved")
    take = np.flatnonzero(result.x > .5) if result.x is not None else None
    if take is not None and not np.array_equal(groups[take], np.arange(query.shape[0])):
        raise ValueError("Solver witness did not choose one legal slate per query")
    upper = getattr(result, "mip_dual_bound", None)
    return take, dict(status=status, upper_native=float(-upper * norm)
                      if upper is not None and np.isfinite(upper) else None)
