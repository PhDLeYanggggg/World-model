"""Training-defined support strata and descriptive cost-fit diagnostics."""
import numpy as np
from scipy.stats import spearmanr


def fit_cuts(distance, speed, known):
    d, s, known = np.asarray(distance), np.asarray(speed), np.asarray(known)
    if (d.ndim != 1 or d.shape != s.shape or known.shape != d.shape or known.dtype.kind != 'b'
            or not known.any() or not np.isfinite(d).all() or not np.isfinite(s).all()
            or np.any(d < 0) or np.any(s < 0)):
        raise ValueError('Finite causal training quantities and complete training support required')
    def cut(x):
        positive = x[known & (x > 0)]
        if not len(positive): return None
        return np.quantile(positive, [.5, .9, .99]).tolist()
    return dict(disagreement=cut(d), past_step_displacement=cut(s), quantiles=[.5, .9, .99],
                fit_population='complete_cost_training_positive_values')


def strata(values, boundaries):
    x = np.asarray(values)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Finite nonnegative causal support required')
    if boundaries is None:
        return {'zero':x == 0, 'positive_no_training_support':x > 0}
    b = np.asarray(boundaries)
    if b.shape != (3,) or np.any(np.diff(b) < 0) or not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError('Three ordered positive training quantiles required')
    assigned = np.searchsorted(b, x, side='left')
    out = {'zero':x == 0}
    for i, name in enumerate(('positive_to_q50', 'q50_to_q90', 'q90_to_q99', 'above_q99')):
        out[name] = (x > 0) & (assigned == i)
    assert np.array_equal(sum(out.values()), np.ones(len(x), int))
    return out


def cost_stats(score, target, distance, mask, supported, constant_native, constant_fraction):
    n = len(distance)
    score, target = np.asarray(score), np.asarray(target)
    mask, supported, distance = np.asarray(mask), np.asarray(supported), np.asarray(distance)
    if (score.shape != (n, 2) or target.shape != score.shape or mask.shape != (n,)
            or supported.shape != mask.shape or mask.dtype.kind != 'b' or supported.dtype.kind != 'b'
            or not np.isfinite(score).all() or not np.isfinite(target[supported]).all()
            or not np.isfinite(distance).all() or (distance < 0).any()):
        raise ValueError('Aligned scores, complete costs and explicit support required')
    use = mask & supported
    out = dict(rows=int(mask.sum()), complete_rows=int(use.sum()),
               excluded_incomplete=int((mask & ~supported).sum()))
    if not use.any(): return dict(out, statistics=None)
    p, y, d = score[use], target[use], distance[use]
    den = np.where(d > 0, d, 1.)[:, None]
    error = p-y; predicted_net, actual_net = p[:, 0]-p[:, 1], y[:, 0]-y[:, 1]
    rho = None
    if np.ptp(predicted_net) > 0 and np.ptp(actual_net) > 0:
        rho = float(spearmanr(predicted_net, actual_net).statistic)
    native_constant = np.broadcast_to(np.asarray(constant_native), y.shape).copy()
    native_constant[d == 0] = 0
    fraction_constant = d[:, None]*np.asarray(constant_fraction)
    st = dict(predicted_benefit=float(p[:, 0].mean()), predicted_harm=float(p[:, 1].mean()),
        realized_benefit=float(y[:, 0].mean()), realized_harm=float(y[:, 1].mean()),
        benefit_bias=float(error[:, 0].mean()), harm_bias=float(error[:, 1].mean()),
        native_MSE=float((error**2).mean()), fraction_MSE=float(((error/den)**2).mean()),
        native_constant_MSE=float(((native_constant-y)**2).mean()),
        fraction_constant_MSE=float((((fraction_constant-y)/den)**2).mean()),
        realized_net_gain=float(actual_net.mean()), predicted_net_gain=float(predicted_net.mean()),
        positive_gain_fraction=float((actual_net > 0).mean()), net_spearman=rho,
        predicted_positive_but_harmful_fraction=float(((predicted_net > 0) & (actual_net < 0)).mean()))
    return dict(out, statistics=st)
