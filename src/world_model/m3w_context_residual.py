"""Past-only context summaries and fixed closed-form cost-bias probes."""
import numpy as np

NAMES = ('path_efficiency', 'speed_change', 'turn_angle', 'neighbor_count',
         'nearest_distance', 'closing_speed', 'rollout_disagreement')
ARMS = ('global_bias', 'context_bias')


def features(geometry, width, reference, candidate):
    g, width, r, p = map(lambda a: np.asarray(a, float), (geometry, width, reference, candidate))
    n = len(g)
    if (g.shape != (n, 476) or width.shape != (n,) or r.shape != (n, 12, 2)
            or p.shape != r.shape or any(not np.isfinite(a).all() for a in (g, width, r, p))
            or (width <= 0).any() or not np.isin(g[:, 230:294], [0, 1]).all()):
        raise ValueError('Fixed past geometry, positive current width and causal rollouts required')
    h = g[:, :16].reshape(n, 8, 2)
    v = np.diff(h, axis=1); speed = np.linalg.norm(v, axis=2)
    path = speed.sum(1); scale = np.maximum(path, width)
    velocity_scale = np.maximum(speed.mean(1), width / 7)
    turn = np.arctan2(v[:, :-1, 0]*v[:, 1:, 1]-v[:, :-1, 1]*v[:, 1:, 0],
                     (v[:, :-1]*v[:, 1:]).sum(2))
    turn = np.where((speed[:, :-1] > 0) & (speed[:, 1:] > 0), np.abs(turn), 0)
    neighbor = g[:, 38:166].reshape(n, 8, 8, 2)
    mask = g[:, 230:294].reshape(n, 8, 8).astype(bool)
    distance = np.linalg.norm(neighbor - h[:, None], axis=3)
    current = np.where(mask[:, :, -1], distance[:, :, -1], np.inf)
    nearest = current.argmin(1); ids = np.arange(n)
    near = current[ids, nearest]; near[~np.isfinite(near)] = np.nan
    both = mask[ids, nearest, -1] & mask[ids, nearest, -2]
    closing = np.where(both, distance[ids, nearest, -2]-distance[ids, nearest, -1], np.nan)
    return np.column_stack((np.linalg.norm(h[:, -1]-h[:, 0], axis=1)/scale,
        (speed[:, -1]-speed[:, 0])/velocity_scale, turn.mean(1), mask[:, :, -1].sum(1),
        near/scale, closing/velocity_scale, np.linalg.norm(r-p, axis=2).mean(1)/scale))


def weighted_cut(values, weights):
    use = np.isfinite(values) & (weights > 0)
    if not use.any(): return None
    x, w = values[use], weights[use]; order = np.argsort(x, kind='stable')
    x, w = x[order], w[order]; cumulative = np.cumsum(w)/w.sum()
    return [float(x[min(np.searchsorted(cumulative, q), len(x)-1)]) for q in (1/3, 2/3)]


def bins(context, cuts):
    x = np.asarray(context, float)
    if x.ndim != 2 or x.shape[1] != len(NAMES) or len(cuts) != len(NAMES) or np.isinf(x).any():
        raise ValueError('Aligned context schema; missing is NaN, not infinity')
    out = np.full(x.shape, 3, dtype=int)
    for j, cut in enumerate(cuts):
        if cut is not None:
            valid = np.isfinite(x[:, j]); out[valid, j] = np.searchsorted(cut, x[valid, j], side='left')
    return out


def design(context, cuts, arm):
    if arm not in ARMS: raise ValueError('Unregistered probe arm')
    b = bins(context, cuts)
    if arm == 'global_bias': return np.ones((len(b), 1))
    return np.column_stack((np.ones(len(b)), (b[:, :, None] == np.arange(4)).reshape(len(b), -1)))


def fit(context, base_costs, target, weights, sites, held, *, ridge=.1):
    """In-sample fitting residuals; held locality must be excluded upstream."""
    x, p, y, w, sites = map(np.asarray, (context, base_costs, target, weights, sites))
    if (p.shape != (len(x), 4) or y.shape != p.shape or w.shape != (len(x),)
            or sites.shape != (len(x),) or held in sites or len(set(sites)) != 3
            or not np.isfinite(p).all() or not np.isfinite(w).all() or (w < 0).any()
            or not ridge > 0):
        raise ValueError('Three fitting localities, excluded held locality and aligned costs required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1), ~known) or (w[~known] != 0).any():
        raise ValueError('Unknown labels must have zero fitting weight')
    use = known & (w > 0)
    if not use.any(): raise ValueError('No supported fitting labels')
    w = w[use].astype(float); w /= w.sum(); xx = x[use]
    cuts = [weighted_cut(xx[:, j], w) for j in range(len(NAMES))]
    rms = max(float(np.sqrt(w@(y[use, 3]**2))), 1e-8)
    residual = (y[use, 3]-p[use, 3])/rms
    out = {}
    for arm in ARMS:
        z = design(xx, cuts, arm); penalty = np.eye(z.shape[1])*ridge; penalty[0, 0] = 0
        lhs = z.T@(w[:, None]*z)+penalty; rhs = z.T@(w*residual)
        beta = np.linalg.solve(lhs, rhs)
        np.testing.assert_allclose(lhs@beta, rhs, rtol=1e-9, atol=1e-10)
        out[arm] = dict(cuts=cuts, rms=rms, coefficients=beta.tolist(), ridge=ridge,
            arm=arm, fitting_sites=sorted(set(sites)), known_rows=int(use.sum()),
            fitted_with_in_sample_base_predictions=True, independent_calibration=False,
            weighted_fitting_residual_MSE=float(w@((residual-z@beta)**2)))
    return out


def predict(model, context, base_costs):
    p = np.asarray(base_costs, float)
    if p.shape != (len(context), 4) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Finite nonnegative causal base costs required')
    shift = design(context, model['cuts'], model['arm'])@np.asarray(model['coefficients'])*model['rms']
    result = p.copy(); result[:, 3] = np.clip(p[:, 3]+shift, 0, p[:, 1])
    return result


def context_table(context, cuts, prediction, target, sites, recordings, agents, rms):
    """Residuals are diagnostic labels, never input features or gating labels."""
    y, p = np.asarray(target), np.asarray(prediction)
    known = np.isfinite(y).all(1); b = bins(context, cuts)
    output = []
    for site in sorted(set(sites)):
        rows = known & (sites == site)
        center = float(np.mean((y[rows, 3]-p[rows, 3])/rms)) if rows.any() else 0.
        cells = []
        for j in range(len(NAMES)):
            values = []
            for k in range(4):
                use = rows & (b[:, j] == k)
                tracks = len(np.unique(np.column_stack((recordings[use], agents[use])), axis=0))
                bias = float(np.mean((y[use, 3]-p[use, 3])/rms)-center) if use.any() else None
                values.append(dict(rows=int(use.sum()), tracks=tracks, centered_bias=bias))
            cells.append(values)
        output.append(dict(site=site, cells=cells))
    return output


def repeated_contexts(fitting, held):
    if len(fitting) != 3 or len(held) != 1: raise ValueError('Three fitting and one held locality required')
    out = []
    for j, name in enumerate(NAMES):
        supported = repeated = same = opposite = 0
        for k in range(4):
            rows = [r['cells'][j][k] for r in fitting + held]
            if not all(r['rows'] >= 20 and r['tracks'] >= 10 for r in rows): continue
            supported += 1; v = [r['centered_bias'] for r in rows]
            signs = np.sign(np.where(np.abs(v) <= 1e-8, 0., v))
            if signs[0] != 0 and all(s == signs[0] for s in signs[:3]):
                repeated += 1; same += int(signs[-1] == signs[0]); opposite += int(signs[-1] == -signs[0])
        out.append(dict(feature=name, supported_cells=supported, repeated_fitting_sign=repeated,
                        held_same_sign=same, held_opposite_sign=opposite))
    return out
