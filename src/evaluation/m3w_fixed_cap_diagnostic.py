"""Offline realized-label error accounting, not a prediction or deployment API."""
import numpy as np

from src.world_model.m3w_context_residual import weighted_cut


def fixed_cap_accounting(prediction, target, cap, weights):
    p, y, c, w = (np.asarray(a, dtype=float) for a in (prediction, target, cap, weights))
    if (p.ndim != 1 or any(a.shape != p.shape for a in (y, c, w))
            or any(not np.isfinite(a).all() for a in (p, c, w))
            or np.isinf(y).any() or (p < 0).any() or (c < 0).any()
            or (p > c + 1e-5 * np.maximum(1, c)).any() or (w < 0).any()
            or (y[np.isfinite(y)] < 0).any() or (w[np.isnan(y)] != 0).any()):
        raise ValueError('Aligned nested nonnegative predictions, labels and zero unknown weights required')
    use = np.isfinite(y) & (w > 0)
    if not use.any():
        raise ValueError('Positive fitting support required')
    p, y, c, w = (a[use] for a in (p, y, c, w))
    w = w / w.sum()
    q = np.minimum(y, c)
    floor, distance, cross = (y-q)**2, (p-q)**2, 2*(y-q)*(q-p)
    error = (p-y)**2
    np.testing.assert_allclose(error, floor+distance+cross, rtol=1e-10, atol=1e-10)
    assert min(floor.min(), distance.min(), cross.min()) >= -1e-8
    mse = float(w @ error)
    projection_floor = float(w @ floor)
    return dict(rows=int(use.sum()), mse=mse, projection_floor=projection_floor,
                distance_to_projection=float(w @ distance), boundary_cross_term=float(w @ cross),
                max_identity_error=float(np.max(np.abs(error-floor-distance-cross))),
                floor_share_percent=100*projection_floor/mse if mse > 0 else None,
                fraction_above_cap=float(w @ (y > c)),
                mean_target_minus_cap=float(w @ (y-c)),
                conditional_bias_identified=False, deployable_oracle=False)


def diagnose_bank(prediction, target, envelope, registered_weights):
    p, y, env, rw = (np.asarray(a, dtype=float) for a in (prediction, target, envelope, registered_weights))
    if (env.ndim != 1 or p.shape != (len(env), 4) or y.shape != p.shape or rw.shape != env.shape
            or any(not np.isfinite(a).all() for a in (p, env, rw)) or np.isinf(y).any()
            or (p < 0).any() or (env < 0).any() or (rw < 0).any()):
        raise ValueError('Aligned finite nested costs required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(~known, np.isnan(y).all(1)) or (rw[~known] != 0).any():
        raise ValueError('Unknown cost rows must be wholly unknown and receive zero weight')
    tol = 1e-5*np.maximum(env, 1)
    for a in (p, y[known]):
        ee, tt = (env, tol) if a is p else (env[known], tol[known])
        if ((a < 0).any() or (a[:, 3] > a[:, 1]+tt).any()
                or (a[:, 2] > a[:, 0]+tt).any() or (a[:, 1] > ee+tt).any()):
            raise ValueError('Nested harm and causal envelope ordering violated')
    valid = known & (env > 0)
    score = np.divide(p[:, 1], env, out=np.zeros(len(env)), where=env > 0)
    output = {}
    for name, weights in (('uniform_positive_envelope', valid.astype(float)),
                          ('registered_risk_weights', np.where(valid, rw, 0))):
        use = weights > 0
        if not use.any():
            raise ValueError('Positive fitting support required for both registered summaries')
        cut = weighted_cut(score, weights)
        bins = np.searchsorted(cut, score, side='left')
        cells = []
        for k in range(3):
            take = use & (bins == k)
            if not take.any():
                cells.append(dict(bin=k, rows=0, mean_target_minus_cap=None,
                                  normalized_mean_target_minus_cap=None))
                continue
            w = weights[take]/weights[take].sum()
            gap = float(w @ (y[take, 3]-p[take, 1]))
            rms = float(np.sqrt(w @ (y[take, 3]**2)))
            cells.append(dict(bin=k, rows=int(take.sum()), mean_target_minus_cap=gap,
                              normalized_mean_target_minus_cap=gap/rms if rms > 0 else None))
        output[name] = dict(
            frozen_harm_cap=fixed_cap_accounting(p[:, 3], y[:, 3], p[:, 1], weights),
            causal_envelope_cap=fixed_cap_accounting(p[:, 3], y[:, 3], env, weights),
            risk_bin_cuts=cut, risk_bins=cells,
            known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
            known_zero_envelope_rows=int((known & (env == 0)).sum()))
    return output
