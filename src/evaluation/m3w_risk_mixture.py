"""Descriptive symmetric decomposition of a ratio gap on common support."""

import numpy as np


def gap(source, target, numerator="harm", denominator="reference"):
    """Separate reference-mass mix and within-stratum rate, not causal effects."""
    if set(source) != set(target):
        raise ValueError("Require the same stratum roster")
    bins = sorted(source)
    ref = np.array([[d[k][denominator] for k in bins] for d in (source, target)], float)
    harm = np.array([[d[k][numerator] for k in bins] for d in (source, target)], float)
    if (not np.isfinite(ref).all() or not np.isfinite(harm).all()
            or (ref < 0).any() or (harm < 0).any()):
        raise ValueError("Nonnegative finite masses required")
    common = (ref > 0).all(0)
    if not common.any():
        return dict(status="no_common_positive_reference", strata=[])
    r, h = ref[:, common], harm[:, common]
    weights, rate = r/r.sum(1, keepdims=True), h/r
    pooled = np.sum(weights*rate, axis=1)
    mixture = float((weights[1]-weights[0])@rate.mean(0))
    within = float((rate[1]-rate[0])@weights.mean(0))
    if not np.isclose(pooled[1]-pooled[0], mixture+within, rtol=0., atol=1e-10):
        raise ValueError("Ratio decomposition failed")
    return dict(status="complete" if common.all() else "common_support_only",
                strata=np.array(bins)[common].tolist(), reference_coverage=(r.sum(1)/ref.sum(1)).tolist(),
                source_risk=float(pooled[0]), target_risk=float(pooled[1]), gap=float(pooled[1]-pooled[0]),
                composition=mixture, within=within, reference_weights=weights.tolist(), stratum_rates=rate.tolist())
