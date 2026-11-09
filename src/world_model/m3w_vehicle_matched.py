"""Risk-admitted joint choices and exact-radius matched controls."""

import numpy as np

from src.world_model.m3w_wepdtof_matched_policy import FRACTION

BUDGET = .02
ARMS = ('joint', 'matched_independent', 'matched_family')


def same_radius(cv, bank, offsets):
    delta = bank - cv[:, None]
    peak = np.linalg.norm(delta, axis=-1).max(axis=-1)
    paths = np.broadcast_to(cv[:, None], bank.shape).copy()
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        if hi-lo < 2:
            continue
        local = peak[lo:hi]
        common = local.min(axis=1)
        support = common[common > 1e-8]
        if len(support) < 2:
            continue
        r = FRACTION * np.median(support)
        ok = (local > 1e-8) & (local >= r)
        ratio = np.divide(r, local, out=np.zeros_like(local), where=ok)
        paths[lo:hi] += delta[lo:hi] * ratio[..., None, None]
    return paths


def decide(value, legal, *, max_edits):
    """Return selected row/family pairs; confidence is not fitted on held data."""
    if value.shape != (*legal.shape, 6) or not np.isfinite(value).all() or max_edits not in (0, 1, 2):
        raise ValueError('Bad causal scores or edit budget')
    row, arm = np.nonzero(legal)
    out = {k: np.empty((0, 2), dtype=int) for k in ARMS}
    if max_edits == 0 or not len(row):
        return out
    v = value[row, arm]
    utility, excess = v[:, :2], v[:, 2:4] - BUDGET*v[:, 4:6]
    independent = utility.min(axis=1)
    admissible = np.where((excess <= 0.).all(axis=1), independent, -np.inf)
    j = int(admissible.argmax())
    chosen = [j] if admissible[j] > 0. else []
    score = float(admissible[j]) if chosen else 0.
    a, b = np.triu_indices(len(row), 1)
    keep = row[a] != row[b]
    a, b = a[keep], b[keep]
    same = arm[a] == arm[b]
    # A two-edit comparison is only estimable if the same-family control exists.
    if max_edits == 2 and same.any():
        allowed = ((excess[a] + excess[b]) <= 0.).all(axis=1)
        pair = np.where(allowed, (utility[a]+utility[b]).min(axis=1), -np.inf)
        k = int(pair.argmax())
        if pair[k] > score:
            chosen = [a[k], b[k]]
    if not chosen:
        return out
    if len(chosen) == 1:
        additive = uniform = [int(independent.argmax())]
    else:
        add = independent[a]+independent[b]
        k, m = int(add.argmax()), int(np.where(same, add, -np.inf).argmax())
        additive, uniform = [a[k], b[k]], [a[m], b[m]]
    for name, selected in zip(ARMS, (chosen, additive, uniform), strict=True):
        out[name] = np.column_stack((row[selected], arm[selected]))
    return out


def choose(cv, paths, value, offsets):
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    out = {k: cv.copy() for k in ARMS}
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        decision = decide(value[lo:hi], peak[lo:hi] > 1e-8, max_edits=min(2, hi-lo-1))
        sizes = []
        for name, take in decision.items():
            row, arm = take.T
            out[name][lo+row] = paths[lo+row, arm]
            sizes.append(np.sort(peak[lo+row, arm]))
        for v in sizes[1:]:
            np.testing.assert_allclose(sizes[0], v, atol=1e-9, rtol=1e-9)
    return out
