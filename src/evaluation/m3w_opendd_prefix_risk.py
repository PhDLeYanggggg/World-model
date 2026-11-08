"""Sequential allocation from predicted risk moments, with matched controls."""
from itertools import combinations

import numpy as np


def subsets(eligible, limit):
    if limit<=0:
        return [()]
    ids = np.flatnonzero(eligible)
    return [(), *[(int(j),) for j in ids], *(combinations(map(int, ids), 2) if limit>=2 else ())]


def compression(origin, cv, new, edits):
    pairs = np.triu(np.linalg.norm(origin[:,None]-origin[None],axis=-1)<=15.,k=1)
    at,bt = np.nonzero(pairs)
    if not len(at) or not edits:
        return 0.
    ref = cv + origin[:,None]
    pred = ref.copy()
    rows = list(edits)
    pred[rows] = new[rows] + origin[rows,None]
    return float(np.maximum(0.,np.linalg.norm(ref[at]-ref[bt],axis=-1)
        -np.linalg.norm(pred[at]-pred[bt],axis=-1)).mean())


def choose(moments, eligible, origin, cv, new, credit, *, joint=False):
    """Use only current-query predictions and already reserved moments."""
    best, best_value, next_credit = (),0.,credit
    for ids in subsets(eligible,min(2,len(eligible)-1))[1:]:
        add = moments[list(ids)].sum(0)
        proposed = credit + add[1:]
        if (proposed[2:]<=0.).any() or (proposed[:2]>.02*proposed[2:]+1e-12).any():
            continue
        value = add[0] - (.25*compression(origin,cv,new,ids) if joint else 0.)
        if value > best_value:
            best,best_value,next_credit = ids,float(value),proposed
    return best,next_credit


def control(moments, eligible, origin, cv, new, count, *, additive):
    if count==0:
        return ()
    pool = [r for r in subsets(eligible,min(2,len(eligible)-1)) if len(r)==count]
    single = [.25*compression(origin,cv,new,(i,)) if additive and eligible[i] else 0.
              for i in range(len(eligible))]
    return max(pool,key=lambda ids:sum(moments[i,0]-single[i] for i in ids))


def walk(d, candidate, estimates):
    output = {k:np.zeros(len(candidate),dtype=bool) for k in ('local','prefix','joint','matched_independent','matched_additive')}
    budget = {}
    geo = {k:0. for k in output}
    for lo,hi in zip(d['offsets'][:-1],d['offsets'][1:],strict=True):
        rec = d['recordings'][lo]
        if not (d['recordings'][lo:hi]==rec).all():
            raise ValueError('Query spans different recordings')
        block = slice(lo,hi)
        p,c,o,m = candidate[block],d['cv'][block],d['origin'][block],estimates[block]
        eligible = (p!=c).any(axis=(1,2))
        selected = {}
        for arm in ('local','prefix','joint'):
            credit = budget.get((rec,arm),np.zeros(4)) if arm!='local' else np.zeros(4)
            selected[arm], budget[rec,arm] = choose(m,eligible,o,c,p,credit,joint=arm=='joint')
        for arm in ('matched_independent','matched_additive'):
            selected[arm] = control(m,eligible,o,c,p,len(selected['joint']),additive=arm=='matched_additive')
        for arm,ids in selected.items():
            output[arm][lo+np.asarray(ids,dtype=int)] = True
            geo[arm] += compression(o,c,p,ids)
    return output,geo
