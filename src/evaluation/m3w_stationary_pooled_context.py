"""Low-dimensional, permutation-invariant summaries of the fixed causal probe."""
import numpy as np


NAMES = ['visible_count', 'aligned_count', 'degenerate_scale', 'nearest_distance', 'mean_distance', 'max_distance',
         'mean_path', 'mean_last_speed', 'moving_fraction', 'mean_closing_speed', 'min_closest_time',
         'mean_closest_distance', 'mean_straightness']
GEOMETRY_COLUMNS = list(range(6))


def pool_context(features):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 83 or not np.isfinite(x).all():
        raise ValueError('Fixed finite 83-column causal features required')
    n = x[:, 3:].reshape(-1, 8, 10)
    out = np.zeros((len(x), len(NAMES)))
    out[:, :3] = x[:, :3]
    for i, neighbors in enumerate(n):
        valid = neighbors[neighbors[:, 0] == 1]
        if not len(valid):
            out[i, 10] = 10.
            continue
        out[i, 3:] = [valid[:, 1].min(), valid[:, 1].mean(), valid[:, 1].max(),
                       valid[:, 2].mean(), valid[:, 3].mean(), np.mean(valid[:, 3] > 0),
                       valid[:, 6].mean(), valid[:, 7].min(), valid[:, 8].mean(), valid[:, 9].mean()]
    return out
