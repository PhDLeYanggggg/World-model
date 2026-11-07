"""Past-only speed and heading dynamics with a speed-matched turn control."""
import numpy as np

HALF_LIVES = (3., 8.)
TURN_TRIGGER = np.deg2rad(5.)
SPEED_TRIGGER = .10


def options(history, ols, cv):
    if history.shape[1:] != (8, 2) or ols.shape != cv.shape or cv.shape[1:] != (12, 2):
        raise ValueError('Expected 8 observed and 12 forecast points')
    d = np.diff(history, axis=1)
    speed = np.linalg.norm(d, axis=2)
    v0 = ols[:, 0]
    s0 = np.linalg.norm(v0, axis=1)
    direction = v0/np.maximum(s0[:, None], 1e-8)
    scale = np.maximum(s0, np.median(speed[:, -5:], axis=1))
    accel = np.clip(np.median(np.diff(speed, axis=1)[:, -5:], axis=1), -.25*scale, .25*scale)
    left, right = d[:, :-1], d[:, 1:]
    cross = left[..., 0]*right[..., 1]-left[..., 1]*right[..., 0]
    dot = np.sum(left*right, axis=2)
    valid = (speed[:, :-1] > 1e-6) & (speed[:, 1:] > 1e-6)
    turns = np.where(valid, np.arctan2(cross, dot), 0.)[:, -5:]
    rate = np.median(turns, axis=1)
    coherent = ((np.sign(rate[:, None])*turns) >= TURN_TRIGGER).sum(1) >= 3
    rate = np.where(coherent, np.clip(rate, -np.pi/9., np.pi/9.), 0.)
    speed_signal = (s0 > 1e-6) & (abs(accel) >= SPEED_TRIGGER*scale)
    turn_signal = (s0 > 1e-6) & coherent
    straight, turning = [], []
    for half in HALF_LIVES:
        integral = np.cumsum(2.**(-np.arange(12)/half))
        speeds = np.clip(s0[:, None]+accel[:, None]*integral, 0., 2.*scale[:, None])
        angles = rate[:, None]*integral
        u = direction[:, None]
        perp = np.stack((-direction[:, 1], direction[:, 0]), axis=1)[:, None]
        for seq, orient in ((straight, u), (turning, np.cos(angles)[..., None]*u+np.sin(angles)[..., None]*perp)):
            path = np.cumsum(speeds[..., None]*orient, axis=1)
            identity = np.linalg.norm(path-cv, axis=2).max(1) <= 1e-6
            path[identity] = cv[identity]
            seq.append(path)
    return np.stack(straight, 1), np.stack(turning, 1), speed_signal, turn_signal


def fixed_recipient_paths(original, cv, straight, turning, speed_signal, turn_signal):
    selected = (original != cv).any(axis=(1, 2))
    replace_speed = selected & speed_signal & (straight != cv).any(axis=(1, 2))
    replace_turn = selected & turn_signal & (turning != cv).any(axis=(1, 2))
    speed_path = np.where(replace_speed[:, None, None], straight, original)
    turn_path = np.where(replace_turn[:, None, None], turning, speed_path)
    return speed_path, turn_path
