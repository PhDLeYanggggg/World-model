"""Reference prediction on immutable selected sets; no new risk certificate."""
import numpy as np


def selected_reference(target, known, scale, choice, predictions):
    y, k, s, c = map(np.asarray, (target, known, scale, choice))
    n = len(y)
    if (y.shape != (n,) or k.shape != (n,) or k.dtype != bool
            or s.shape != (n,) or c.shape != (n,) or c.dtype.kind not in 'iu'
            or np.any((c < -1) | (c >= 7)) or not np.isfinite(y).all()
            or np.any(y < 0) or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError('Aligned labels, causal scale and immutable choices required')
    selected = c >= 0
    use = selected & k
    support = bool(np.any(use & (y > 0)))
    normalizer = float(s[use].sum())
    result = dict(selected=int(selected.sum()), selected_known=int(use.sum()),
        selected_unknown=int((selected & ~k).sum()), actual_known_ER_mass=float(y[use].sum()),
        observed_reference_support=support if selected.any() else None, models={})
    for name, value in predictions.items():
        p = np.asarray(value)
        if p.shape != (n,) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Finite aligned nonnegative reference predictions required')
        error = float((p[use].sum() - y[use].sum()) / normalizer) if normalizer > 0 else None
        result['models'][name] = dict(
            predicted_full_ER_mass=float(p[selected].sum()),
            predicted_known_ER_mass=float(p[use].sum()),
            predicted_unknown_ER_credit=float(p[selected & ~k].sum()),
            known_row_normalized_MSE=float(np.mean(((p[use] - y[use]) / s[use]) ** 2)) if use.any() else None,
            known_record_normalized_mass_error=error,
            known_record_normalized_mass_squared_error=error ** 2 if error is not None else None,
            positive_prediction_without_observed_support=bool(selected.any() and not support and p[selected].sum() > 0))
    return result
