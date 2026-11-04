"""Past-only image-plane common-motion controls; no camera-pose claim."""
import numpy as np

from src.world_model.m3w_european_source_forecast import BASELINES, baseline_numpy

EXTRA = tuple(dict(name=f'{family}_acceleration_a{a:g}_d{d:g}', family=family, alpha=a, decay=d)
    for family in ('nearest8', 'all_current') for a in (.25, .5, 1.) for d in (.8, .9, 1.))
CANDIDATES = tuple(dict(name=name, family='original', index=k) for k,name in enumerate(BASELINES))+EXTRA


def accelerations(history, geometry, shared_context):
    h, g, s = map(np.asarray, (history, geometry, shared_context))
    n = len(h)
    if (h.shape != (n, 8, 2) or g.shape != (n, 476) or s.shape != (n, 48)
            or not all(np.isfinite(v).all() for v in (h,g,s))):
        raise ValueError('Original complete past/nearest8 geometry and causal shared48 required')
    local = np.empty((n, 2))
    for start in range(0, n, 1024):
        stop = min(n, start+1024)
        neighbors = g[start:stop, 38:166].reshape(-1, 8, 8, 2)
        valid = g[start:stop, 230:294].reshape(-1, 8, 8).all(-1)
        own = np.diff(h[start:stop], axis=1)[:, None]
        velocities = np.concatenate((own, np.diff(neighbors, axis=2)), axis=1)
        mask = np.column_stack((np.ones(stop-start, bool), valid))
        common = np.nanmedian(np.where(mask[:, :, None, None], velocities, np.nan), axis=1)
        local[start:stop] = (common[:, -1]-common[:, -4])/3
    scale = np.maximum(np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1), 1.)
    common = s[:, :42].reshape(n, 7, 6)[:, :, 2:4]*scale[:, None, None]
    global_acceleration = (common[:, -1]-common[:, -4])/3
    return dict(nearest8=local, all_current=global_acceleration)


def rollout(history, acceleration, candidate):
    h, a = np.asarray(history), np.asarray(acceleration)
    if h.shape != (len(h), 8, 2) or not np.isfinite(h).all():
        raise ValueError('Finite eight-step causal history required')
    if candidate['family'] == 'original':
        return baseline_numpy(h, candidate['index'])-h[:, -1, None]
    if a.shape != (len(h), 2) or not np.isfinite(a).all() or candidate not in CANDIDATES:
        raise ValueError('Registered causal common acceleration and fixed candidate required')
    t = np.arange(1, 13)
    displacement = np.asarray([sum((k-j+1)*candidate['decay']**j for j in range(1, k+1)) for k in t])
    return (h[:, -1]-h[:, -2])[:, None]*t[None, :, None]+candidate['alpha']*a[:, None]*displacement[None, :, None]


def fitting_choice(costs, sites, indices):
    values, names = np.asarray(costs), np.asarray(sites)
    if values.shape != (len(names), len(CANDIDATES)) or 1 not in indices:
        raise ValueError('Complete fixed candidate matrix on fitting rows only required')
    scores = []
    for site in sorted(set(names)):
        use = (names == site)&np.isfinite(values[:, 1])
        if not use.any() or not np.isfinite(values[use]).all():
            raise ValueError('Common known-label fitting support required')
        mean = values[use].mean(0)
        if mean[1] <= 0:
            raise ValueError('Positive fitting CV reference required')
        scores.append(mean/mean[1])
    score = np.mean(scores, 0)
    return min(indices, key=lambda k:(score[k],k)), score.tolist()
