"""Shared-roster pair geometry scores and paired bounds with partial futures."""
import numpy as np

from src.world_model.m3w_record_joint_actions import local_edges

ARM_FIELDS = ('pair_ade', 'pair_fde', 'pair_contraction', 'scene_ade')
PAIR_FIELDS = ('observed_ade_reduction', 'grid_ade_lower', 'grid_ade_upper',
               'observed_fde_reduction', 'grid_fde_lower', 'grid_fde_upper')


def query(history, paths, target, valid, comparisons):
    """Positive contrasts favor the right arm; unobserved pairs are bounded."""
    i, j = local_edges(history)
    observed = valid[i] & valid[j]
    terminal = observed[:, -1]
    clean = np.where(valid[..., None], target, 0.)
    dy = clean[i]-clean[j]
    p = paths[:, i]-paths[:, j]
    err = np.linalg.norm(p-dy[None], axis=-1)
    origin = (history[i, -1]-history[j, -1])[:, None]
    distance = np.linalg.norm(dy+origin, axis=-1)
    predicted_distance = np.linalg.norm(p+origin[None], axis=-1)
    contraction = np.maximum(distance[None]-predicted_distance, 0.)
    scene = np.linalg.norm(paths-clean[None], axis=-1)
    missing = ~observed

    def masked(a, mask):
        return a[:, mask].mean(1) if mask.any() else np.full(len(paths), np.nan)

    arms = np.column_stack((masked(err, observed), masked(err[:, :, -1], terminal),
                            masked(contraction, observed), masked(scene, valid)))
    pairs = np.full((len(comparisons), len(PAIR_FIELDS)), np.nan)
    changed = np.zeros(len(comparisons), bool)
    if not len(i):
        return arms, pairs, changed, (0, 0, 0, 0)
    for k, (a, b) in enumerate(comparisons):
        gap = err[a]-err[b]
        width = np.linalg.norm(p[a]-p[b], axis=-1)
        known_sum = float(gap[observed].sum())
        unknown_width = float(width[missing].sum())
        final_known = float(gap[terminal, -1].sum())
        final_unknown = float(width[~terminal, -1].sum())
        pairs[k] = [known_sum/observed.sum() if observed.any() else np.nan,
                    (known_sum-unknown_width)/observed.size,
                    (known_sum+unknown_width)/observed.size,
                    final_known/terminal.sum() if terminal.any() else np.nan,
                    (final_known-final_unknown)/len(i), (final_known+final_unknown)/len(i)]
        changed[k] = bool((width > 0.).any())
    return arms, pairs, changed, (observed.size, int(observed.sum()), len(i), int(terminal.sum()))


def mean_finite(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    count = finite.sum(0)
    return np.divide(np.where(finite, values, 0.).sum(0), count,
                     out=np.full(values.shape[1:], np.nan), where=count > 0)


def bootstrap(cells, draw):
    """Cells are [source, seed]; do not drop unsupported cells."""
    data = np.asarray(cells, dtype=float)
    if not np.isfinite(data).all():
        return None
    values = data.mean(1)
    return dict(point=float(values.mean()), interval95=np.quantile(values[draw].mean(1), [.025, .975]).tolist())
