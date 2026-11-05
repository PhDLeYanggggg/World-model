"""Causal directional nearest-peer path profiles, not observed future motion."""
import numpy as np

NAMES = tuple(f'{path}_nearest_step{t}_{axis}' for path in ('floor', 'candidate')
    for t in range(1, 13) for axis in ('forward', 'side')) + tuple(
    f'candidate_minus_floor_nearest_radius_step{t}' for t in range(1, 13))


def basis(x):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (7, 428) or not np.isfinite(x).all():
        raise ValueError('Finite causal428 only; no future-label interface')
    if not (x[..., 24:32] == 1).all():
        raise ValueError('Eight complete past states required')
    h = x[..., :24].reshape(*x.shape[:2], 8, 3).astype(float)
    n = x[..., 32:224].reshape(*x.shape[:2], 8, 8, 3).astype(float)
    mask = x[..., 224:288].reshape(*x.shape[:2], 8, 8)
    if not np.isin(mask, (0, 1)).all():
        raise ValueError('Binary neighbor validity required')
    valid = (mask[..., -2:] == 1).all(-1) & (n[..., -2:, 2] == h[..., None, -2:, 2]).all(-1)
    velocity = np.diff(h[..., :2], axis=-2)
    moving = np.linalg.norm(velocity, axis=-1) > 1e-12
    index = np.where(moving, np.arange(7), -1).max(-1)
    direction = np.take_along_axis(velocity, index.clip(0)[..., None, None], axis=-2)[..., 0, :]
    length = np.linalg.norm(direction, axis=-1)
    direction = np.divide(direction, length[..., None], out=np.zeros_like(direction), where=length[..., None] > 1e-12)
    direction = np.where((index >= 0)[..., None], direction, np.array([1., 0.]))
    pos = np.where(valid[..., None], n[..., -1, :2]-h[..., None, -1, :2], 0)
    vel = np.where(valid[..., None], n[..., -1, :2]-n[..., -2, :2], 0)
    peer = pos[..., None, :]+np.arange(1, 13)[None, :, None]*vel[..., None, :]
    parts, distances = [], []
    for start in (306, 330):
        ego = x[..., start:start+24].reshape(*x.shape[:2], 12, 2).astype(float)
        relative = peer-ego[..., None, :, :]
        radius = np.linalg.norm(relative, axis=-1)
        minimum = np.where(valid[..., None], radius, np.inf).min(-2)
        # Averaging exact ties makes nearest-peer identity permutation invariant.
        nearest = valid[..., None] & (radius == minimum[..., None, :])
        vector = (relative*nearest[..., None]).sum(-3)/nearest.sum(-2).clip(1)[..., None]
        forward = (vector*direction[..., None, :]).sum(-1)
        side = direction[..., None, 0]*vector[..., 1]-direction[..., None, 1]*vector[..., 0]
        parts.append(np.stack((forward, side), -1).reshape(*x.shape[:2], 24))
        distances.append(np.where(valid.any(-1)[..., None], minimum, 0))
    out = np.concatenate((*parts, distances[1]-distances[0]), -1).astype(np.float32)
    if out.shape != (*x.shape[:2], 60) or not np.isfinite(out).all():
        raise ValueError('Finite causal peer-path60 required')
    return out


def chunked(x, chunk=512):
    return np.concatenate([basis(x[i:i+chunk]) for i in range(0, len(x), chunk)])


def misaligned(features, groups, ids):
    if features.shape != (len(ids), 7, 60) or len(set(ids)) != len(ids):
        raise ValueError('Unique aligned causal row identities required')
    seen = np.zeros(len(ids), int); result = np.empty_like(features); donors = np.full(len(ids), -1, int)
    for group in groups:
        at = np.asarray(group, int)
        if not len(at) or (at < 0).any() or (at >= len(ids)).any():
            raise ValueError('Nonempty valid complete query required')
        ordered = at[np.argsort(ids[at], kind='stable')]
        donor = np.roll(ordered, 1)
        result[ordered] = features[donor]; donors[ordered] = donor; np.add.at(seen, at, 1)
    if not (seen == 1).all():
        raise ValueError('Every row must appear in exactly one query')
    return result, donors
