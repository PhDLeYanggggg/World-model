"""Past-only coupled rotation choices for two fixed vehicle paths."""

import numpy as np

from src.evaluation.m3w_angular_incumbent_oracle import rotate


DEGREES = (0, *(a for a in range(-90, 91, 15) if a))
CAUSAL = ('shared', 'independent', 'coupled')


def candidates(cv, anchor):
    if cv.shape != anchor.shape or cv.ndim != 3 or len(cv) != 2:
        raise ValueError('Expected two fixed and aligned vehicle paths')
    a = np.deg2rad(DEGREES)
    bank = np.stack([rotate(cv, anchor, theta) for theta in a], axis=1)
    bank[:, 0] = anchor
    r = np.linalg.norm(anchor-cv, axis=-1)
    np.testing.assert_allclose(np.linalg.norm(bank-cv[:, None], axis=-1),
                               np.broadcast_to(r[:, None], bank.shape[:-1]), atol=1e-10, rtol=1e-10)
    return bank


def choices(bank, ols):
    """All causal arms force two rotations; only the angle constraint changes."""
    z = bank[:, 1:]
    unary = np.linalg.norm(z-ols[:, None], axis=-1).mean(2)
    u = unary[0, :, None]+unary[1, None, :]
    pair = (z[0, :, None]-z[1, None, :])-(ols[0]-ols[1])
    joint = u+np.linalg.norm(pair, axis=-1).mean(2)
    shared = int(np.diag(u).argmin())+1
    decode = lambda score: tuple(int(i)+1 for i in np.unravel_index(score.argmin(), score.shape))
    return dict(shared=(shared, shared), independent=decode(u), coupled=decode(joint))
