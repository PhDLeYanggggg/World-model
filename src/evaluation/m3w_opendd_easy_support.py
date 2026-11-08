"""Predictive easy-support prioritization without changing the safety test."""
import numpy as np

from src.evaluation.m3w_opendd_prefix_risk import choose, subsets


def event(easy, legal, baseline, candidate):
    return easy & legal & (np.maximum(candidate-baseline,0.)<=.02*baseline)


def revise(ordinary, moments, legal, support, accumulated, allow_insert):
    if accumulated>=1.:
        return ordinary
    if not ordinary and not allow_insert:
        return ordinary
    n = max(1,len(ordinary))
    allowed = []
    for ids in subsets(legal,min(2,len(legal)-1)):
        if len(ids)!=n:
            continue
        add = moments[list(ids)].sum(0)
        if (add[3:]<=0.).any() or (add[1:3]>.02*add[3:]+1e-12).any():
            continue
        p = float(support[list(ids)].sum())
        if not ordinary and p<.25:
            continue
        allowed.append((ids,p,float(add[0])))
    return max(allowed,key=lambda a:(a[1],a[2]),default=(ordinary,0.,0.))[0]


def walk(d, p, moments, prob):
    names = ('local','mass_matched','event_matched','support_first')
    out = {name:np.zeros(len(p),dtype=bool) for name in names}
    credit = {}
    mass = np.divide(moments[:,4],moments[:,3],out=np.zeros(len(p)),where=moments[:,3]>0).clip(0.,1.)
    for a,b in zip(d['offsets'][:-1],d['offsets'][1:],strict=True):
        block = slice(a,b)
        rec = d['recordings'][a]
        cv = d['cv'][block]
        legal = (p[block]!=cv).any(axis=(1,2))
        m = moments[block]
        ordinary,_ = choose(m,legal,d['origin'][block],cv,p[block],np.zeros(4))
        out['local'][a+np.asarray(ordinary,dtype=int)] = True
        for name,score in (('mass_matched',mass),('event_matched',prob),('support_first',prob)):
            key = (rec,name)
            have = credit.get(key,0.)
            ids = revise(ordinary,m,legal,score[block],have,name=='support_first')
            rows = a+np.asarray(ids,dtype=int)
            out[name][rows] = True
            credit[key] = have+score[rows].sum()
    return out
