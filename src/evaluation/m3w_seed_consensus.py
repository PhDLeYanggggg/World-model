"""Fixed seed aggregation of causal risk scores, not calibrated coverage."""
import numpy as np


def scores(predictions):
    p = np.asarray(predictions)
    if p.ndim != 3 or p.shape[0] != 3 or p.shape[2] != 2 or not np.isfinite(p).all():
        raise ValueError('Three aligned finite causal two-channel risk predictions required')
    return dict(mean=p.mean(0), unanimity=p.max(0))
