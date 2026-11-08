"""Causal subset choice with expiring, recording-local edit credits."""

from dataclasses import dataclass

import numpy as np

from src.world_model import m3w_opendd_credit_value as queue


GEOMETRY_WIDTH, QUEUE_WIDTH = 10, 7


def remove_geometry(x):
    x[:, -(GEOMETRY_WIDTH+QUEUE_WIDTH):-QUEUE_WIDTH] = 0.
    return x


@dataclass
class Packet:
    rows: tuple
    features: np.ndarray
    fixed: np.ndarray
    size: np.ndarray
    release: int
    scale: float


def menu(legal, rankings, cap):
    groups, fixed = [np.empty(0, dtype=int)], [0]
    ordered = [legal[np.argsort(-np.asarray(s)[legal], kind="stable")] for s in rankings]
    for k in range(1, cap+1):
        fixed.append(len(groups))
        seen = set()
        for rank in ordered:
            ids = tuple(sorted(rank[:k].tolist()))
            if ids not in seen:
                groups.append(np.asarray(ids))
                seen.add(ids)
    return tuple(groups), np.asarray(fixed)


def geometry(h, cv, ego, options, scale):
    """Relative-rollout summaries, including selected/unselected edges."""
    i, j = np.triu_indices(len(h), 1)
    left = h[:, -1:, :]+cv
    right = h[:, -1:, :]+ego
    sep = lambda z: np.linalg.norm(z[i]-z[j], axis=-1)/scale
    base = sep(left)
    out = []
    for rows in options:
        edited = np.zeros(len(h), dtype=bool)
        edited[rows] = True
        near = sep(np.where(edited[:, None, None], right, left))
        cols = []
        for edges in (edited[i] ^ edited[j], edited[i] & edited[j]):
            a, b = base[edges], near[edges]
            if len(a):
                x, y = a.min(1), b.min(1)
                cols.extend((np.log1p(len(a)), np.log1p(x.min()), np.log1p(y.min()),
                             (y-x).clip(-10., 10.).mean(),
                             (np.log1p(b[:, -1])-np.log1p(a[:, -1])).mean()))
            else:
                cols.extend([0.]*5)
        out.append(cols)
    return np.asarray(out, dtype=np.float32)


def actions(p, state, use_geometry=True):
    steps = list(queue.actions(state, p.release, len(p.fixed)-1))
    available = np.flatnonzero(p.size < len(steps))
    z = np.array([np.log1p([*state, p.release, k, *n, lost]) for k, n, lost in steps], dtype=np.float32)
    x = np.c_[p.features[available], z[p.size[available]]]
    if not use_geometry:
        remove_geometry(x)
    return available, x


def training(packets, reward):
    all_reward = [np.array([reward[r].sum() for r in p.rows]) for p in packets]
    envelopes = [np.array([v[p.size == k].max() for k in range(len(p.fixed))])
                 for p, v in zip(packets, all_reward, strict=True)]
    labels, _ = queue.action_targets([p.release for p in packets], envelopes, [p.scale for p in packets])
    for p, states, value, envelope in zip(packets, labels, all_reward, envelopes, strict=True):
        for state, target in states.items():
            if len(target) <= 1:
                continue
            idx, x = actions(p, state)
            k = p.size[idx[1:]]
            correction = (value[idx[1:]]-envelope[k])/p.scale
            # Slate identity has no effect on the queue beyond cardinality.
            yield x[1:], (target[k]+correction[:, None]).clip(-12., 12.)


def stream(packets, score, arm):
    """Pick slate and count without viewing any later query or end flag."""
    state = (0, 0)
    pending, logs, expired = [], [], 0
    for t, p in enumerate(packets):
        idx, x = actions(p, state, use_geometry=arm != "no_geometry")
        val = score(x)
        if val.shape != (len(idx), 2) or not np.isfinite(val).all():
            raise ValueError("Invalid action score")
        val[0] = 0.
        if arm == "fixed_now":
            pick = p.fixed[p.release]
        elif arm == "adaptive_now":
            use = np.flatnonzero(p.size[idx] == p.release)
            pick = idx[use[val[use, 0].argmax()]]
        else:
            pool = np.flatnonzero(np.isin(idx, p.fixed)) if arm == "fixed_queue" else np.arange(len(idx))
            pick = idx[pool[val[pool, int(arm != "myopic_queue")].argmax()]]
        take = p.size[pick]
        pending.extend((t, int(r)) for r in p.rows[p.fixed[p.release]])
        used = pending[:take]
        _, state, lost = list(queue.actions(state, p.release, len(p.fixed)-1))[take]
        pending = [(s, r) for s, r in pending[take:] if t-s < 2]
        if sum(state) != len(pending):
            raise ValueError("Queue and credit identities disagree")
        expired += lost
        logs.append((p.rows[pick], p.rows[p.fixed[take]], used))
    return logs, expired+len(pending)
