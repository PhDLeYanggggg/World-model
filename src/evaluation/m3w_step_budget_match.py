"""Match the full time profile of an edit without changing its directions."""

import numpy as np


def match(delta):
    """Return original, peak-shrunk, and step-matched two-arm displacement arrays."""
    if delta.shape != (2, 2, 12, 2) or not np.isfinite(delta).all():
        raise ValueError('Expected two finite two-recipient displacement sequences')
    speed = np.linalg.norm(delta, axis=-1)
    maxima = speed.max(axis=-1)
    peak = float(maxima[0, 0])
    if peak <= 0. or not np.allclose(maxima, peak, atol=1e-9, rtol=1e-9):
        raise ValueError('The original edit peaks are not matched')
    radius = speed.min(axis=(0, 1))
    common_peak = float(radius.max())
    if common_peak <= 0.:
        return None
    scale = common_peak/peak
    units = np.divide(delta, speed[..., None], out=np.zeros_like(delta), where=speed[..., None] > 0.)
    output = dict(original=delta, smaller_peak=delta*scale, matched_steps=units*radius[None, None, :, None])
    np.testing.assert_allclose(np.linalg.norm(output['matched_steps'], axis=-1),
                               np.broadcast_to(radius, speed.shape), atol=1e-12, rtol=1e-12)
    return output, radius, scale
