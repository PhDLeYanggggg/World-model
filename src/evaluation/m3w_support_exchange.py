"""Allocate radius by support-preserving transfers around a fixed-half blend."""
import numpy as np

ARMS = ('hindcast','utility','guarded','matched_hindcast')


def labels(cv,y,valid,inc,scale,cut):
    delta = inc-cv
    half = cv+.5*delta
    err = np.linalg.norm(cv-y,axis=-1)
    half_err = np.linalg.norm(half-y,axis=-1)
    known = valid.any(1)
    count = valid.sum(1).clip(1)
    ref = np.where(valid,err,0.).sum(1)/count
    easy = known & (ref>0) & (ref<=cut)
    half_ade = np.where(valid,half_err,0.).sum(1)/count
    half_harm = np.maximum(half_ade-ref,0.)
    out = []
    for level in (.25,.75):
        p = cv+level*delta
        e = np.linalg.norm(p-y,axis=-1)
        step_disagreement = np.linalg.norm(p-half,axis=-1)
        lower = np.where(valid,half_err-e,-step_disagreement).mean(1)
        ade = np.where(valid,e,0.).sum(1)/count
        diff = np.maximum(ade-ref,0.)-half_harm
        debit = (~known)*(level-.5)*np.linalg.norm(delta,axis=-1).max(1)
        out.append(np.column_stack((lower,diff+debit,np.where(easy,diff,0.)+debit,
                                   np.where(easy,ade-half_ade,0.))))
    return np.stack(out,axis=1)/scale[:,None,None]


def options(radius):
    i,j = np.nonzero(np.triu(np.ones((len(radius),len(radius)),dtype=bool),k=1))
    i,j = np.r_[i,j],np.r_[j,i]
    amount = .25*np.minimum(radius[i],radius[j])
    keep = amount>0
    return i[keep],j[keep],amount[keep]


def pick(r,m,h):
    i,j,amount = options(r)
    if not len(i):
        return {a:None for a in ARMS},np.zeros(4)
    contrasts = (amount/(.25*r[i]))[:,None]*m[i,1] + (amount/(.25*r[j]))[:,None]*m[j,0]
    heuristic = amount*(h[i]-h[j])
    safe = (contrasts[:,1:]<=0.).all(1)
    ids = {}
    for name,score in (('hindcast',heuristic),('utility',contrasts[:,0]),
                       ('guarded',np.where(safe,contrasts[:,0],-np.inf))):
        k = int(score.argmax())
        ids[name] = k if score[k]>0. else None
    g = ids['guarded']
    out = {a:None if k is None else (int(i[k]),int(j[k]),float(amount[k])) for a,k in ids.items()}
    out['matched_hindcast'] = None
    if g is not None:
        budget = amount[g]
        k = int(np.where(amount>=budget, h[i]-h[j],-np.inf).argmax())
        out['matched_hindcast'] = int(i[k]),int(j[k]),float(budget)
    return out, contrasts[g] if g is not None else np.zeros(4)


def rollout(cv,inc,offsets,estimates,heuristic):
    r = np.linalg.norm(inc-cv,axis=-1).max(1)
    scale = {a:np.full(len(r),.5) for a in ARMS}
    active = {a:0 for a in ARMS}
    transfer = {a:0. for a in ARMS}
    changes = np.zeros(4)
    for lo,hi in zip(offsets[:-1],offsets[1:],strict=True):
        edits,values = pick(r[lo:hi],estimates[lo:hi],heuristic[lo:hi])
        changes += values
        for name,edit in edits.items():
            if edit is None:
                continue
            i,j,a = edit
            scale[name][lo+i] += a/r[lo+i]
            scale[name][lo+j] -= a/r[lo+j]
            active[name] += 1
            transfer[name] += a
    if active['guarded']!=active['matched_hindcast'] or transfer['guarded']!=transfer['matched_hindcast']:
        raise ValueError('Matched control lost transferred radius')
    paths = {a:cv+v[:,None,None]*(inc-cv) for a,v in scale.items()}
    for p in paths.values():
        dist = np.linalg.norm(p-cv,axis=-1).max(1)
        np.testing.assert_array_equal(dist>0.,r>0.)
        np.testing.assert_allclose(np.add.reduceat(dist,offsets[:-1]),
            .5*np.add.reduceat(r,offsets[:-1]),rtol=1e-7,atol=1e-14)
    return paths,dict(active_queries=active,transferred_radius=transfer,guarded_predicted_changes=changes.tolist())
