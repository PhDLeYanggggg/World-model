"""Dual-reference routing with matched query counts and peak displacement."""
from itertools import combinations, product

import numpy as np

from src.world_model.m3w_reference_bridge_policy import BUDGET

FRACTION = .25
PAIR_WEIGHT = .10
NAMES = ('joint','independent','uniform','no_pair')


def bank(cv,ols,neural,queries):
    offsets = np.stack((ols-cv,neural-cv),axis=1)
    radii = np.linalg.norm(offsets,axis=-1).max(2)
    paths = np.broadcast_to(cv[:,None],offsets.shape).copy()
    for q in np.unique(queries):
        at = np.flatnonzero(queries==q)
        common = radii[at].min(1)
        positive = common[common>1e-8]
        if len(positive)<2:
            continue
        radius = FRACTION*np.median(positive)
        legal = at[common>=radius]
        paths[legal] += offsets[legal]*(radius/radii[legal])[...,None,None]
        np.testing.assert_allclose(np.linalg.norm(paths[legal]-cv[legal,None],axis=-1).max(2),
                                   radius,rtol=1e-8,atol=1e-8)
    return paths


def analytic(h):
    last = h[:,-1:]
    v = last-h[:,-2:-1]
    o = np.einsum('t,ntd->nd',np.array([-1.5,-.5,.5,1.5]),h[:,-4:]-last)[:,None]/5.
    t = np.arange(1.,13.)[None,:,None]
    return v*t,o*t


def choose(est,legal,q,h,cv,paths):
    picks = {k:np.full(len(q),-1,dtype=int) for k in NAMES}
    raw = est[...,:2]
    risk = est[...,2:4]-BUDGET*est[...,4:6]
    for j in np.unique(q):
        ix = np.flatnonzero(q==j)
        eligible = ix[legal[ix].all(1)]
        cap = min(2,len(ix)-1,len(eligible))
        if cap<1:
            continue
        motion = np.linalg.norm(np.diff(h[ix],axis=1),axis=-1)
        local = max(float(np.median(motion))*12.,1e-8)
        near = {}
        for r,s in combinations(eligible,2):
            near[r,s] = np.exp(-np.linalg.norm(h[r,-1]-h[s,-1])/local)
        best, no_pair = (0.,()),{}
        for n in range(1,cap+1):
            for recipients in combinations(eligible,n):
                for arms in product(range(paths.shape[1]),repeat=n):
                    actions = tuple(zip(recipients,arms))
                    a,b = np.array(actions).T
                    if (risk[a,b].sum(0)>0.).any():
                        continue
                    utility = float(raw[a,b].sum(0).min())
                    if n not in no_pair or utility>no_pair[n][0]:
                        no_pair[n] = utility,actions
                    if n==2:
                        offset = paths[a,b]-cv[a]
                        cost = near[recipients]*np.linalg.norm(offset[0]-offset[1],axis=1).mean()
                        utility -= PAIR_WEIGHT*cost
                    if utility>best[0]:
                        best = utility,actions
        selected = best[1]
        n = len(selected)
        if not n:
            continue
        point = raw[eligible].min(2)
        arg = point.argmax(1)
        independent = np.argsort(-point.max(1),kind='stable')[:n]
        choices = []
        for k in range(paths.shape[1]):
            at = np.argsort(-point[:,k],kind='stable')[:n]
            choices.append((point[at,k].sum(),tuple((int(eligible[v]),k) for v in at)))
        control = max(choices,key=lambda v:v[0])[1]
        for k,actions in dict(joint=selected,no_pair=no_pair[n][1],
                independent=tuple(zip(eligible[independent],arg[independent])),uniform=control).items():
            for a,b in actions:
                picks[k][a]=b
    return picks


def apply(cv,paths,picks,q):
    result = {}
    _,idx = np.unique(q,return_inverse=True)
    for name,choice in picks.items():
        edited = choice>=0
        z = cv.copy()
        z[edited] = paths[edited,choice[edited]]
        peak = np.linalg.norm(z-cv,axis=-1).max(1)
        np.testing.assert_array_equal(peak>0.,edited)
        n = np.bincount(idx[edited],minlength=len(np.unique(idx)))
        r = np.bincount(idx,weights=peak,minlength=len(n))
        if result:
            np.testing.assert_array_equal(n,old_n)
            np.testing.assert_allclose(r,old_r,atol=1e-8,rtol=1e-8)
        else:
            old_n,old_r = n,r
        result[name] = z
    return result
