"""One-factor residual target transport with a frozen prediction producer."""
import numpy as np
from src.world_model import m3w_nested_residual as nested
from src.world_model import m3w_context_residual as residual


def common_event_target(raw, cv, sites, outer, cut):
    nested.fitting_sites(sites, outer)
    if len(raw) != len(sites) or len(cv) != len(sites) or not np.isfinite(cut) or cut <= 0:
        raise ValueError('Aligned fitting-only rows and positive outer-fitting cut required')
    return nested.labels(raw, cv, cut)


def raw_shift(model, context):
    return (residual.design(context, model['cuts'], model['arm'])
            @ np.asarray(model['coefficients']) * model['rms'])


def transport_accounting(old, new, context, inner_y, common_y, weights, held_context, frozen):
    """Linear pre-clip event contribution; clipping is reported separately."""
    if old['cuts'] != new['cuts'] or old['ridge'] != new['ridge'] or old['arm'] != new['arm']:
        raise ValueError('Only supervised target semantics may differ')
    known = np.isfinite(inner_y).all(1)
    if not np.array_equal(known, np.isfinite(common_y).all(1)):
        raise ValueError('Unchanged label support required')
    use = known & (weights > 0)
    w = np.asarray(weights[use], float); w /= w.sum()
    z = residual.design(context[use], old['cuts'], old['arm'])
    penalty = np.eye(z.shape[1]) * old['ridge']; penalty[0, 0] = 0
    event = inner_y[use, 3] - common_y[use, 3]
    event_beta = np.linalg.solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w * event))
    coefficient_delta = np.asarray(old['coefficients']) * old['rms'] - np.asarray(new['coefficients']) * new['rms']
    np.testing.assert_allclose(coefficient_delta, event_beta, rtol=1e-7, atol=1e-8)
    old_shift, new_shift = raw_shift(old, held_context), raw_shift(new, held_context)
    predicted_event = residual.design(held_context, old['cuts'], old['arm']) @ event_beta
    np.testing.assert_allclose(old_shift - new_shift, predicted_event, rtol=1e-7, atol=1e-8)
    applied = residual.predict(old, held_context, frozen)[:, 3] - residual.predict(new, held_context, frozen)[:, 3]
    return dict(fitting_event_label_MSE=float(w @ (event**2)),
                coefficient_identity_max_error=float(np.max(np.abs(coefficient_delta-event_beta))),
                held_preclip_event_RMS=float(np.sqrt(np.mean(predicted_event**2))),
                held_postclip_change_RMS=float(np.sqrt(np.mean(applied**2))),
                clipping_difference_RMS=float(np.sqrt(np.mean((applied-predicted_event)**2))),
                outer_held_labels_used=False, causal_effect_proven=False)
