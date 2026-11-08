"""Isolate turn persistence with a shared past-only tangent and speed."""

import numpy as np

from src.world_model import m3w_radius_matched_turn_menu as legacy

NAMES = ("straight", "tapered", "sustained")


def rotate(v, angle):
    x, y = np.moveaxis(v, -1, 0)
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((x*c-y*s, x*s+y*c), axis=-1)


def paths(history, steps):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1] < 4 or h.shape[-1] != 2 or steps < 1 or not np.isfinite(h).all():
        raise ValueError("Require at least four finite 2D observations and a positive horizon")
    v = np.diff(h[:, -4:], axis=1)
    speed = np.linalg.norm(v, axis=-1)
    left, right = v[:, :-1], v[:, 1:]
    cross = left[..., 0]*right[..., 1]-left[..., 1]*right[..., 0]
    angle = np.arctan2(cross, (left*right).sum(-1))
    # If either recent chord pair is stationary, do not extrapolate a turn.
    yaw = np.where((speed > 1e-8).all(1), np.median(angle, axis=1), 0.).clip(-.20, .20)
    aligned = rotate(v, yaw[:, None]*np.arange(2, -1, -1)[None])
    mean = aligned.mean(1)
    tangent = mean/np.linalg.norm(mean, axis=1).clip(1e-8)[:, None]
    initial = tangent*np.median(speed, axis=1)[:, None]
    return np.stack([rotate(initial[:, None], yaw[:, None]*phase[None]).cumsum(1) for phase in
                     (np.zeros(steps), np.cumsum(.8**np.arange(steps)), np.arange(1, steps+1))], axis=1)


def matched(history, steps, amp):
    cv, ols, bank = legacy.make_menu(history, steps, amp)
    offset = paths(history, steps)-cv[:, None]
    r = legacy.peak(bank[:, 0]-cv)
    raw = legacy.peak(offset)
    f = np.divide(r[:, None], raw, out=np.zeros_like(raw), where=raw > 1e-12)
    extra = cv[:, None]+offset*f[..., None, None]
    extra = np.where((raw <= 1e-12)[..., None, None], bank[:, :1], extra)
    np.testing.assert_allclose(legacy.peak(extra-cv[:, None]), np.broadcast_to(r[:, None], raw.shape), atol=1e-10)
    return cv, ols, bank, extra


def choose(history, amplitude):
    """Add one candidate at a time; choose with the same observed-prefix score."""
    cv, _, bank, extra = matched(history, 12, amplitude)
    votes = []
    for k in (4, 5, 6):
        c, _, prev, add = matched(history[:, :k], 8-k, amplitude)
        y = history[:, k:]-history[:, k-1:k]
        scale = np.linalg.norm(np.diff(history[:, :k], axis=1), axis=-1).mean(1).clip(.05)
        ref = np.linalg.norm(c-y, axis=-1).mean(1)
        e = np.linalg.norm(np.concatenate((prev, add), axis=1)-y[:, None], axis=-1).mean(2)
        votes.append((ref[:, None]-e)/scale[:, None]/(8-k))
    v = np.mean(votes, axis=0)
    row = np.arange(len(cv))
    best = v[:, :bank.shape[1]].argmax(1)
    anchor = bank[row, best]
    used = v[:, bank.shape[1]:] > v[row, best, None]
    selected = np.where(used[..., None, None], extra, anchor[:, None])
    return anchor, selected, extra, used
