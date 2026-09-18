"""Training-only equal-episode exposure, using past-defined group identities."""
import numpy as np


def episode_weights(ids, all_ids, episode_keys):
    ids, all_ids, episode_keys = map(np.asarray, (ids, all_ids, episode_keys))
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or all_ids.ndim != 1 or episode_keys.shape != all_ids.shape
            or np.any(np.diff(all_ids) <= 0)):
        raise ValueError('Unique training IDs and sorted source mapping required')
    pos = np.searchsorted(all_ids, ids)
    if np.any(pos == len(all_ids)) or not np.array_equal(all_ids[pos], ids):
        raise ValueError('Training row missing from source episode mapping')
    groups, inv, counts = np.unique(episode_keys[pos], return_inverse=True, return_counts=True)
    weights = 1. / (len(groups)*counts[inv])
    assert (weights > 0).all() and np.isclose(weights.sum(), 1)
    return weights, dict(training_episodes=len(groups), training_rows=len(ids),
                        smallest_row_probability=float(weights.min()),
                        largest_row_probability=float(weights.max()))
