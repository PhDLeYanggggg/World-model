"""Past-derived neighbor CV versus causal forecast paths, not collision labels."""
import numpy as np

PATH_NAMES = ('minimum', 'mean_nearest', 'endpoint_nearest', 'maximum_approach_fraction',
    'mean_approach_fraction', 'minimum_time_fraction', 'minimum_to_current', 'endpoint_to_current')
NAMES = ('neighbor_support', 'neighbor_mean_speed', 'current_minimum', 'current_mean') + tuple(
    f'{p}_{name}' for p in ('floor', 'candidate') for name in PATH_NAMES) + tuple(
    'candidate_minus_floor_'+name for name in PATH_NAMES[:4])


def basis(x):
    """No outcomes, label masks or endpoints are accepted by this API."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (7, 428) or not np.isfinite(x).all():
        raise ValueError('Exact finite causal428 schema required')
    if not (x[..., 24:32] == 1).all():
        raise ValueError('Complete eight-step ego history required')
    h = x[..., :24].reshape(*x.shape[:2], 8, 3).astype(float)
    n = x[..., 32:224].reshape(*x.shape[:2], 8, 8, 3).astype(float)
    mask = x[..., 224:288].reshape(*x.shape[:2], 8, 8)
    if not np.isin(mask, (0, 1)).all():
        raise ValueError('Binary neighbor masks required')
    aligned = (n[..., -2:, 2] == h[..., None, -2:, 2]).all(-1)
    valid = (mask[..., -2:] == 1).all(-1) & aligned
    count = valid.sum(-1); supported = count > 0
    pos = np.where(valid[..., None], n[..., -1, :2]-h[..., None, -1, :2], 0)
    vel = np.where(valid[..., None], n[..., -1, :2]-n[..., -2, :2], 0)
    radius = np.linalg.norm(pos, axis=-1)
    minimum = np.where(supported, np.where(valid, radius, np.inf).min(-1), 0)
    base = np.stack((count/8, (np.linalg.norm(vel, axis=-1)*valid).sum(-1)/count.clip(1),
        minimum, (radius*valid).sum(-1)/count.clip(1)), -1)
    forecast = pos[..., None, :]+np.arange(1, 13)[None, :, None]*vel[..., None, :]
    paths = []
    for start in (306, 330):
        ego = x[..., start:start+24].reshape(*x.shape[:2], 12, 2).astype(float)
        distance = np.linalg.norm(forecast-ego[..., None, :, :], axis=-1)
        nearest = np.where(valid[..., None], distance, np.inf).min(-2)
        nearest = np.where(supported[..., None], nearest, 0)
        approach = ((distance < radius[..., None]) & valid[..., None]).sum(-2)/count.clip(1)[..., None]
        closest = nearest.min(-1); endpoint = nearest[..., -1]
        ratio = lambda a: np.divide(a, minimum, out=np.zeros_like(a), where=minimum > 1e-12)
        paths.append(np.stack((closest, nearest.mean(-1), endpoint, approach.max(-1), approach.mean(-1),
            np.where(supported, (nearest.argmin(-1)+1)/12, 0), ratio(closest), ratio(endpoint)), -1))
    out = np.concatenate((base, *paths, paths[1][..., :4]-paths[0][..., :4]), -1).astype(np.float32)
    if out.shape != (*x.shape[:2], 24) or not np.isfinite(out).all():
        raise ValueError('Finite causal neighbor24 required')
    return out


def chunked(x, chunk=512):
    return np.concatenate([basis(x[i:i+chunk]) for i in range(0, len(x), chunk)])
