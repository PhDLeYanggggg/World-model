"""Fixed rotations of relative predictions; not new forecasts for deployment."""
import numpy as np


def rotation_nulls(prediction):
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (12, 2) or not np.isfinite(p).all():
        raise ValueError('Finite complete relative trajectories required')
    return dict(original=p.copy(), plus90=np.stack((-p[...,1],p[...,0]),axis=-1),
                minus90=np.stack((p[...,1],-p[...,0]),axis=-1), reverse=-p)


def rotated_costs(prediction, target):
    target = np.asarray(target, dtype=np.float64)
    variants = rotation_nulls(prediction)
    if target.shape != variants['original'].shape or not np.isfinite(target).all():
        raise ValueError('Aligned finite evaluation targets required')
    return {name:np.linalg.norm(p-target,axis=-1).mean(1) for name,p in variants.items()}
