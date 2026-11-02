"""Weighted cost-support projection; no targets or policy thresholds accepted."""
import numpy as np


def _pair(a,b,va,vb,d):
    lagrange = np.maximum(0.,(a+b-d)/(va+vb))
    qa,qb = np.maximum(0.,a-lagrange*va),np.maximum(0.,b-lagrange*vb)
    qa = np.where(qb <= 0,np.minimum(a,d),qa)
    qb = np.where(qa <= 0,np.minimum(b,d),qb)
    qa = np.minimum(qa,d)
    qb = np.minimum(qb,np.maximum(0.,d-qa))
    return qa,qb


def project(prediction,actual_radius,rms):
    p,d,r = map(lambda a:np.asarray(a,dtype=float),(prediction,actual_radius,rms))
    if (p.ndim != 3 or p.shape[1:] != (7,5) or d.shape != p.shape[:2] or r.shape != (5,)
            or not all(np.isfinite(a).all() for a in (p,d,r)) or (p < 0).any()
            or (d < 0).any() or (r <= 0).any() or (p[:,:,4] > p[:,:,1]).any()
            or (p[:,:,0] > p[:,:,2]).any() or (p[:,:,3] > p[:,:,2]).any()):
        raise ValueError('Finite coherent native costs, causal radius and frozen positive RMS required')
    vb,vh,ve = r[[0,1,4]]**2
    b,h,e = p[:,:,0],p[:,:,1],p[:,:,4]
    qb,qh = _pair(b,h,vb,vh,d)
    active = qh < e
    combined_variance = vh*ve/(vh+ve)
    combined_mean = (ve*h+vh*e)/(vh+ve)
    ab,ah = _pair(b,combined_mean,vb,combined_variance,d)
    out = p.copy()
    out[:,:,0] = np.where(active,ab,qb)
    out[:,:,1] = np.where(active,ah,qh)
    out[:,:,4] = np.where(active,ah,e)
    return out
