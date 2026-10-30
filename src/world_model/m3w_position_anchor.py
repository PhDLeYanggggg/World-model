"""Change only the causal OLS position intercept,retaining the velocity."""
import numpy as np
from src.world_model.m3w_european_source_forecast import baseline_numpy

WINDOWS = (4, 8)
ANCHORS = ('raw_last', 'causal_fitted')


def predict(history, *, window, anchor):
    h = np.asarray(history, dtype=np.float64)
    if (h.ndim != 3 or h.shape[1:] != (8, 2) or not len(h) or not np.isfinite(h).all()
            or window not in WINDOWS or anchor not in ANCHORS):
        raise ValueError('Complete finite8-step past,OLS4/8 and registered position anchor required')
    raw = baseline_numpy(h, 4 if window == 4 else 5)
    if anchor == 'raw_last': return raw
    x = np.arange(window, dtype=np.float64)-(window-1)/2
    velocity = np.sum(h[:, -window:]*x[None, :, None], axis=1)/np.sum(x*x)
    fitted_current = h[:, -window:].mean(1)+(window-1)/2*velocity
    result = raw+(fitted_current-h[:, -1])[:, None]
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite causal position prediction')
    return result
