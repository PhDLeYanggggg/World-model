"""Past-only single-agent and nearest-approach trajectory alternatives."""

import numpy as np

OWN = ("decaying_acceleration", "late_velocity_smoothing")
NEIGHBOR = ("approach_brake", "approach_lateral")


def candidates(history, steps):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all() or steps != 12:
        raise ValueError("Require eight finite observed positions and twelve forecast steps")
    d = np.diff(h, axis=1)
    v = d[:, -1]
    speed = np.linalg.norm(v, axis=1)
    t = np.arange(1, steps+1)
    cv = v[:, None] * t[None, :, None]
    a = (d[:, -2:].mean(1)-d[:, -4:-2].mean(1))/2.
    accel_phase = np.cumsum(np.cumsum(.72 ** (t-1)))
    smooth_phase = np.cumsum(1.-.8 ** t)
    own = np.stack((cv+a[:, None]*accel_phase[None, :, None],
                    cv+(d[:, -6:].mean(1)-v)[:, None]*smooth_phase[None, :, None]), axis=1)

    # Find threats in a comoving frame, without any fixed metric distance.
    r = h[None, :, -1] - h[:, None, -1]
    u = v[None, :] - v[:, None]
    uu = np.square(u).sum(2)
    tau = np.divide(-(r*u).sum(2), uu, out=np.zeros_like(uu), where=uu > 1e-18)
    closest = r + tau[..., None]*u
    travel = speed[None, :] + speed[:, None]
    miss = np.linalg.norm(closest, axis=-1)
    eligible = (tau > 0.) & (tau <= steps) & (miss <= 2.5*travel) & (speed[:, None] > 1e-9)
    score = np.where(eligible, miss/np.maximum(travel, 1e-9)+.05*tau, np.inf)
    social = np.broadcast_to(cv[:, None], (len(h), 2, steps, 2)).copy()
    if len(h) < 2:
        return cv, own, social, np.full(len(h), -1, dtype=int)
    partner = score.argmin(1)
    rows = np.arange(len(h))
    active = np.isfinite(score[rows, partner])
    ids = rows[active]
    other = partner[active]
    start = np.maximum(tau[ids, other]*.5 - 1., 0.)
    phase = np.cumsum(1.-.70**np.maximum(t[None]-start[:, None], 0.), axis=1)
    tangent = v[ids]/speed[ids, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    side = (normal*closest[ids, other]).sum(1)
    side = np.where(np.abs(side) > 1e-9, side, (normal*r[ids, other]).sum(1))
    social[ids, 0] -= (.45*phase*speed[ids, None])[..., None]*tangent[:, None]
    social[ids, 1] -= (.35*phase*speed[ids, None]*np.sign(side)[:, None])[..., None]*normal[:, None]
    return cv, own, social, np.where(active, partner, -1)
