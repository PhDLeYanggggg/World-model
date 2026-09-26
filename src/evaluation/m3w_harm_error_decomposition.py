"""Diagnostic additive error attribution, not label-aware model selection."""
import numpy as np


def error_decomposition(mean, fractional, target, envelope, training_edge):
    known = np.isfinite(target).all(1) & (envelope > 0)
    if not known.any():
        return dict(status='not_estimable')
    y, old, new, e = target[known, 3], mean[known, 3], fractional[known, 3], envelope[known]
    a, b = (old-y)**2, (new-y)**2
    parts = {}
    masks = dict(zero_harm=y == 0, positive_harm=y > 0,
                 below_training_edge=e <= training_edge, above_training_edge=e > training_edge)
    for name, mask in masks.items():
        parts[name] = dict(rows=int(mask.sum()),
                          mean_MSE_contribution=float(a[mask].sum()/len(y)),
                          fractional_MSE_contribution=float(b[mask].sum()/len(y)),
                          excess_MSE_contribution=float((b[mask]-a[mask]).sum()/len(y)))
    delta = float((b-a).mean())
    for first, second in (('zero_harm', 'positive_harm'), ('below_training_edge', 'above_training_edge')):
        np.testing.assert_allclose(sum(parts[k]['excess_MSE_contribution'] for k in (first, second)),
                                   delta, rtol=1e-10, atol=1e-10)
    return dict(rows=len(y), mean_MSE=float(a.mean()), fractional_MSE=float(b.mean()),
                excess_MSE=delta, training_edge=float(training_edge), parts=parts)
