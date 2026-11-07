"""Context matching using only the first observed query of a recording."""
import hashlib

import numpy as np

from src.world_model.m3w_release_guard import choose_threshold


def profile(history):
    if history.ndim!=3 or history.shape[1:]!=(8,2) or len(history)==0 or not np.isfinite(history).all():
        raise ValueError('Require a nonempty finite eight-observation query')
    steps = np.linalg.norm(np.diff(history,axis=1),axis=-1)
    speed = np.median(steps,axis=1)
    length = steps.sum(1)
    displacement = np.linalg.norm(history[:,-1]-history[:,0],axis=-1)
    straight = np.divide(displacement,length,out=np.ones_like(length),where=length>0.)
    origin = history[:,-1]
    dist = np.linalg.norm(origin[:,None]-origin[None],axis=-1)
    np.fill_diagonal(dist,np.inf)
    nn = dist.min(1)
    good = (nn>0.)&np.isfinite(nn)
    motion = speed[speed>0.]
    scale = np.median(nn[good]) if good.any() else np.median(motion) if len(motion) else 1.
    ratio = speed/scale
    return np.r_[np.log1p(len(history)),np.log1p(np.quantile(ratio,[.25,.5,.75])),
                 np.quantile(straight,[.25,.75]),np.mean(ratio<=1e-6),float(good.any())]


def fitting_scale(fit_profiles):
    x = np.stack(fit_profiles)
    return np.maximum(.10,np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0))


def peers(target,profiles,by_source,scale,kind):
    picked = []
    for names in by_source.values():
        if not names:
            raise ValueError('Every calibration source needs a recording')
        def distance(r):
            return float(np.linalg.norm((target-profiles[r])/scale))
        if kind=='near':
            r = min(names,key=lambda q:(distance(q),q))
        elif kind=='far':
            r = min(names,key=lambda q:(-distance(q),q))
        elif kind=='blind':
            r = min(names,key=lambda q:hashlib.sha256(q.encode()).hexdigest())
        else:
            raise ValueError('Unknown calibration comparison')
        picked.append((r,distance(r)))
    return picked


def threshold(rows):
    candidates = []
    for v in zip(*rows):
        pp = [r['lower_pp'] for r in v]
        candidates.append(dict(preserved=all(r['preserved'] for r in v),
            lower_pp=None if any(x is None for x in pp) else float(np.mean(pp))))
    return choose_threshold(candidates)
