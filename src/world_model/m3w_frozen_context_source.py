"""Bind frozen latent extraction to its forecaster's original causal packet."""
import numpy as np


def original_context(controller, packed):
    names = {'sites', 'recordings', 'frames', 'agents', 'history', 'origin'}
    if set(controller) != names | {'geometry', 'detector_history'} or set(packed) != names | {'geometry'}:
        raise ValueError('Only explicit past context and row identities are accepted')
    for name in names:
        np.testing.assert_array_equal(controller[name], packed[name])
    a, b = controller['geometry'], packed['geometry']
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 476 or a.dtype != b.dtype or b.dtype != np.float32:
        raise ValueError('Original float32 476-column forecaster packet required')
    for start in range(0, len(a), 4096):
        for lo, hi in ((0, 24), (332, 356)):
            np.testing.assert_array_equal(a[start:start+4096, lo:hi], b[start:start+4096, lo:hi])
    return dict(controller, geometry=b)
