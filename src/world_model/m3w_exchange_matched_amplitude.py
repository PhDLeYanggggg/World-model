"""Move displacement between forecast steps without changing its total."""

import numpy as np


def matched_amplitude_paths(cv, prediction):
    cv, prediction = np.asarray(cv, dtype=float), np.asarray(prediction, dtype=float)
    if (cv.shape != prediction.shape or cv.ndim != 3 or cv.shape[-1] != 2 or cv.shape[1] < 4
            or not np.isfinite(cv).all() or not np.isfinite(prediction).all()):
        raise ValueError('Expected finite paired forecast paths')
    residual = prediction-cv
    a = np.linalg.norm(residual, axis=-1)
    radius = a.max(1, keepdims=True)
    mutable = (a > 0.) & (a < radius)
    mutable[:, -1] = False
    t = np.arange(a.shape[1])
    mid = len(t)//2
    left, right = mutable & (t < mid), mutable & (t >= mid)
    taper = np.r_[np.sin(np.pi*(t[:mid]+1)/(mid+1))**2,
                  np.sin(np.pi*(t[mid:]-mid+1)/(len(t)-mid+1))**2]
    paths = {'original': prediction}
    moved = {}
    for name, donor, receiver in (('early', right, left), ('late', left, right)):
        give = a*taper*donor
        take = (radius-a)*taper*receiver
        g, r = give.sum(1), take.sum(1)
        mass = .5*np.minimum(g, r)
        new = a-give*np.divide(mass, g, out=np.zeros_like(g), where=g > 0.)[:, None]
        new += take*np.divide(mass, r, out=np.zeros_like(r), where=r > 0.)[:, None]
        relative = np.divide(new, a, out=np.ones_like(a), where=a > 0.)
        paths[name] = np.where(mutable[..., None], cv+residual*relative[..., None], prediction)
        moved[name] = mass
    return paths, moved
