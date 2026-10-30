"""Label-side squared-error accounting,not a causal model or new metric gate."""
import numpy as np


def decompose(raw, fitted, target, valid, history, scales):
    p, q, y, h, s = map(lambda x: np.asarray(x, float), (raw, fitted, target, history, scales))
    m = np.asarray(valid)
    if (p.ndim != 3 or p.shape[1:] != (12, 2) or q.shape != p.shape or y.shape != p.shape
            or m.shape != p.shape[:2] or m.dtype != bool or h.shape != (len(p), 8, 2)
            or s.shape != (len(p),) or (s <= 0).any() or not all(np.isfinite(x).all() for x in (p, q, h, s))
            or not np.isfinite(y[m]).all()): raise ValueError('Aligned predictions,masked labels and frozen causal scales required')
    d = (q-p)/s[:, None, None]
    r = np.where(m[..., None], (p-y)/s[:, None, None], 0.)
    f = np.where(m[..., None], (q-y)/s[:, None, None], 0.)
    cross = 2*np.sum(r*d, -1); shift = np.sum(d*d, -1)
    raw_sq = np.sum(r*r, -1); fitted_sq = np.sum(f*f, -1)
    direct = fitted_sq-raw_sq; residual = direct-cross-shift
    # A floating-point identity check,not a tolerance on any risk inequality.
    bound = 128*np.finfo(float).eps*(raw_sq+fitted_sq+shift)
    if np.any(np.abs(residual[m]) > bound[m]): raise ArithmeticError('Squared-error identity failed')
    count = m.sum(1); known = count > 0
    def row_mean(x):
        return np.divide(np.where(m, x, 0).sum(1), count, out=np.full(len(p), np.nan), where=known)
    terms = {k: row_mean(v) for k, v in dict(direct=direct, cross=cross, squared_shift=shift).items()}
    acc = h[:, -1]-2*h[:, -2]+h[:, -3]
    alignment = np.sum((q[:, 0]-p[:, 0])*acc, 1)/s**2
    active = (np.linalg.norm(acc, axis=1) > 0) & (np.linalg.norm(q[:, 0]-p[:, 0], axis=1) > 0)
    return dict(rows=len(p), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        observed_points=int(m.sum()), terms={k: float(v[known].mean()) if known.any() else None for k, v in terms.items()},
        squared_error_increased_rows=int((terms['direct'][known] > 0).sum()),
        max_abs_identity_error=float(np.max(np.abs(residual[m]))) if m.any() else None,
        max_abs_constant_shift_rounding=float(np.max(np.abs(d-d[:, :1]))),
        past_acceleration_alignment=dict(active_rows=int(active.sum()), opposite_direction_rows=int((alignment[active] < 0).sum()),
            mean_scaled_dot=float(alignment[active].mean()) if active.any() else None),
        per_step=[dict(observed_rows=int(m[:, t].sum()),
            **{k: float(v[m[:, t], t].mean()) if m[:, t].any() else None
               for k, v in dict(direct=direct, cross=cross, squared_shift=shift).items()}) for t in range(12)],
        future_labels_diagnostic_only=True, loss='available_step_mean_squared_Euclidean_error_divided_by_frozen_scale_squared',
        not_ADE_primary_or_physical_noise_identification=True)
