"""Causal producer-bundle tags with dimension-matched controls."""
import numpy as np

ARMS = ('global', 'producer', 'placebo')


def producer_roles(sites, fitting_ids, held_ids, halves):
    sites = np.asarray(sites); fit, held = np.asarray(fitting_ids), np.asarray(held_ids)
    a, b = map(set, halves)
    if len(a) != 2 or len(b) != 2 or a & b or set(sites[fit]) != a | b or (a | b) & set(sites[held]):
        raise ValueError('Two excluded producer halves within four fitting sources required')
    if set(fit) & set(held) or len(set(fit)) != len(fit) or len(set(held)) != len(held):
        raise ValueError('Distinct aligned fitting/readout indices required')
    # Rows from one half were predicted by the other half, never by themselves.
    tag = np.where(np.isin(sites[fit], sorted(b)), 0, 1).astype(np.int8)
    for side in (0, 1):
        if set(sites[fit[tag == side]]) & set(halves[side]):
            raise ValueError('Producer trained on its target rows')
    return tag


def placebo_tag(ids):
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < 0).any():
        raise ValueError('Nonnegative causal row keys required')
    # Fixed integer mixing is only a negative control; no outcome or locality input.
    x = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((x ^ (x >> np.uint64(31))) & np.uint64(1)).astype(np.int8)


def augment(x, producer, ids, arm):
    x, producer, ids = np.asarray(x), np.asarray(producer), np.asarray(ids)
    if (arm not in ARMS or x.ndim != 2 or x.shape[1] != 380 or not np.isfinite(x).all()
            or producer.shape != (len(x),) or ids.shape != (len(x),) or not np.isin(producer, (0, 1)).all()):
        raise ValueError('Registered causal feature schema and producer identities required')
    tags = np.zeros((len(x), 2), np.float32)
    if arm != 'global':
        tag = producer if arm == 'producer' else placebo_tag(ids)
        tags[np.arange(len(x)), tag.astype(int)] = 1
    return np.column_stack((x, tags)).astype(np.float32)


def safe_choice(utility, risk, latest_moving, budget=.02):
    utility, risk = np.asarray(utility), np.asarray(risk)
    moving = np.asarray(latest_moving)
    if (utility.shape != risk.shape or utility.shape != (len(moving), 2) or moving.dtype != bool
            or not np.isfinite(utility).all() or not np.isfinite(risk).all()
            or (utility < 0).any() or (risk < 0).any() or budget != .02):
        raise ValueError('Finite nonnegative moments, causal stop mask and fixed budget required')
    return moving & (utility[:, 0] > utility[:, 1]) & (risk[:, 1] <= budget*risk[:, 0])


def independent_choice(utility, risk, moving):
    return np.array([bool(m) and float(u[0]) > float(u[1]) and float(r[1]) <= .02*float(r[0])
                     for u, r, m in zip(utility, risk, moving)], bool)
