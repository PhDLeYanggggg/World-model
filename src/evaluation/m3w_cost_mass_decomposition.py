"""Same-set cost masses: separate predicted harm from reference/support error."""
import numpy as np

MOMENTS = ('B', 'H', 'R', 'ER', 'EH')


def cost_masses(prediction, truth, take, alpha=None):
    p, y, use = np.asarray(prediction), np.asarray(truth), np.asarray(take)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 5 or use.shape != (len(p),) or use.dtype != bool:
        raise ValueError('Aligned five-moment rows and explicit boolean set required')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError('Observed/conservative target contributions must be finite and nonnegative')
    a = np.ones(len(p)) if alpha is None else np.asarray(alpha)
    if a.shape != (len(p),) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError('Future-free bounded amplitude required')
    weights = np.ones_like(p); weights[:, [0, 1, 4]] = a[:, None]
    weighted_p, weighted_y = p*weights, y*weights
    out = {}
    for k, name in enumerate(MOMENTS):
        x, z = weighted_p[use, k], weighted_y[use, k]
        pred, target = float(x.sum()), float(z.sum())
        out[name] = dict(predicted_mass=pred, target_mass=target,
            predicted_over_target=pred/target if target > 0 else None,
            mean_absolute_error=float(np.abs(x-z).mean()) if len(x) else None)
    return out


def risks(masses):
    """These are ratios on exactly one selected set, not marginal error scores."""
    def ratio(n, d, field):
        a, b = masses[n][field], masses[d][field]
        return a/b if b > 0 else None
    return dict(predicted_all=ratio('H', 'R', 'predicted_mass'),
        target_all=ratio('H', 'R', 'target_mass'),
        predicted_easy=ratio('EH', 'ER', 'predicted_mass'),
        target_easy=ratio('EH', 'ER', 'target_mass'))
