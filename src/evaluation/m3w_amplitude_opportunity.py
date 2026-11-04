"""Fixed past-only interpolation candidates, separate from label-aware diagnosis."""
import numpy as np

FRACTIONS=(1/32,1/16,1/8,1/4,1/2,1.)


def rollout(floor,proposal,alpha):
    b,p=np.asarray(floor),np.asarray(proposal)
    if (b.ndim != 3 or b.shape[-2:] != (12,2) or p.shape != b.shape
            or not np.isfinite(b).all() or not np.isfinite(p).all()
            or alpha not in (0.,*FRACTIONS)):
        raise ValueError('Two finite past-derived12-step paths and a fixed amplitude required')
    if alpha == 0:
        return b.copy()
    if alpha == 1:
        return p.copy()
    return b+alpha*(p-b)
