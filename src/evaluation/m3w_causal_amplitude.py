"""Causal variable-radius allocation and exact-budget query controls."""
from collections import deque

import numpy as np

LEVELS = np.array([.25, .5, 1.])


def prefix_features(history, offsets, recordings, incumbent):
    """Describe the causal observed footprint without any labels or future rows."""
    z = np.zeros((len(history), 6))
    blocks = {}
    seen = {}
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        rec = recordings[lo]
        if not (recordings[lo:hi]==rec).all():
            raise ValueError('A query spans different recordings')
        past = blocks.setdefault(rec,deque(maxlen=8))
        now = history[lo:hi]
        cloud = np.concatenate([*past,now.reshape(-1,2)])
        low, high = cloud.min(0), cloud.max(0)
        scale = np.linalg.norm(high-low).clip(.1)
        endpoints = now[:, -1]+incumbent[lo:hi, -1]
        outside = np.maximum(np.maximum(low-endpoints,endpoints-high),0.)
        center = cloud.mean(0)
        speed = np.linalg.norm(np.diff(now,axis=1),axis=-1).mean(1)
        heading = now[:,-1]-now[:,0]
        outward = (heading*(now[:,-1]-center)).sum(1)/(
            np.linalg.norm(heading,axis=1)*scale).clip(1e-9)
        z[lo:hi] = np.column_stack((np.full(len(now),np.log1p(seen.get(rec,0))),
            np.full(len(now),np.log1p(len(cloud))),np.linalg.norm(outside,axis=1)/scale,
            np.linalg.norm(endpoints-center,axis=1)/scale,speed/scale,outward))
        seen[rec] = seen.get(rec,0)+1
        past.append(now.reshape(-1,2).copy())
    return z


def geometry(origin, cv, bank):
    """Exact near-pair compression for singleton and two-edit slates."""
    i, j = np.nonzero(np.triu(np.linalg.norm(origin[:,None]-origin[None],axis=-1)<=15.,k=1))
    n, k = bank.shape[:2]
    if not len(i):
        return np.zeros((n,k)),np.zeros((n,n,k,k))
    ref = cv+origin[:,None]
    pred = bank+origin[:,None,None]
    dist = np.linalg.norm(ref[i]-ref[j],axis=-1)
    ai = np.maximum(dist[:,None]-np.linalg.norm(pred[i]-ref[j,None],axis=-1),0.).mean(2)
    aj = np.maximum(dist[:,None]-np.linalg.norm(ref[i,None]-pred[j],axis=-1),0.).mean(2)
    both = np.maximum(dist[:,None,None]-np.linalg.norm(
        pred[i,:,None]-pred[j,None,:],axis=-1),0.).mean(3)
    single = np.zeros((n,k))
    np.add.at(single,i,ai)
    np.add.at(single,j,aj)
    interaction = np.zeros((n,n,k,k))
    interaction[i,j] = both-ai[:,:,None]-aj[:,None,:]
    interaction[j,i] = interaction[i,j].transpose(0,2,1)
    return single/len(i),interaction/len(i)


def choose(moments, eligible, credit, geom=None):
    """Maximize predicted lower utility under the current predicted ledger."""
    i, a = np.nonzero(np.broadcast_to(eligible[:,None],moments.shape[:2]))
    if not len(i):
        return (),credit
    values = moments[i,a].copy()
    if geom is not None:
        single, interaction = geom
        values[:,0] -= .25*single[i,a]
    first, second = np.triu_indices(len(i),k=1)
    keep = (i[first]!=i[second]) & (len(eligible)>2)
    first, second = first[keep],second[keep]
    pair = values[first]+values[second]
    if geom is not None:
        pair[:,0] -= .25*interaction[i[first],i[second],a[first],a[second]]
    options = np.concatenate((values,pair))
    updated = credit+options[:,1:]
    ok = (updated[:,2:]>0).all(1) & (updated[:,:2]<=.02*updated[:,2:]+1e-12).all(1)
    utility = np.where(ok,options[:,0],-np.inf)
    pick = int(utility.argmax())
    if utility[pick]<=0.:
        return (),credit
    ids = (pick,) if pick<len(i) else (first[pick-len(i)],second[pick-len(i)])
    return tuple((int(i[j]),int(a[j])) for j in ids),updated[pick]


def matched(origin, cv, incumbent, radii, radii_wanted, utility):
    """Reassign the same radii exactly; two distinct recipients when needed."""
    if len(radii_wanted)==0:
        return {},{}
    levels = np.r_[0.,LEVELS]
    desired = np.asarray(radii_wanted)
    fractions = np.divide(desired[None],radii[:,None],out=np.zeros((len(radii),len(desired))),
                          where=radii[:,None]>0)
    feasible = (radii[:,None]>0) & (fractions<=1.+1e-12)
    est = np.array([np.interp(x,levels,np.r_[0.,u]) for x,u in zip(fractions,utility,strict=True)])
    bank = cv[:,None]+fractions.clip(0.,1.)[...,None,None]*(incumbent-cv)[:,None]
    single, _ = geometry(origin,cv,bank)
    options = []
    for penalty in (0.,.25):
        z = np.where(feasible,est-penalty*single,-np.inf)
        if len(desired)==1:
            chosen = [int(z[:,0].argmax())]
            score = z[chosen[0],0]
        else:
            score_matrix = z[:,0,None]+z[None,:,1]
            np.fill_diagonal(score_matrix,-np.inf)
            chosen = np.unravel_index(score_matrix.argmax(),score_matrix.shape)
            score = score_matrix[chosen]
        if not np.isfinite(score):
            raise ValueError('Original joint edit cannot be matched')
        options.append({int(i):bank[i,j] for j,i in enumerate(chosen)})
    return tuple(options)


def rollout(cv, incumbent, origin, offsets, recordings, predictions):
    """Run online; inputs are causal arrays and model predictions only."""
    names = ('direct','factor_local','factor_prefix','joint','matched_independent','matched_additive')
    paths = {k:cv.copy() for k in names}
    radius = np.linalg.norm(incumbent-cv,axis=-1).max(1)
    banks = cv[:,None]+LEVELS[None,:,None,None]*(incumbent-cv)[:,None]
    ledger = {}
    for lo,hi in zip(offsets[:-1],offsets[1:],strict=True):
        rec = recordings[lo]
        if not (recordings[lo:hi]==rec).all():
            raise ValueError('A query spans different recordings')
        sl = slice(lo,hi)
        g = geometry(origin[sl],cv[sl],banks[sl])
        want = []
        for arm in names[:4]:
            key = 'factor_prefix' if arm=='joint' else arm
            edits, updated = choose(predictions[key][sl],radius[sl]>0.,
                ledger.get((rec,arm),np.zeros(4)),g if arm=='joint' else None)
            ledger[rec,arm] = updated
            for i,a in edits:
                paths[arm][lo+i] = banks[lo+i,a]
            if arm=='joint':
                want = [radius[lo+i]*LEVELS[a] for i,a in edits]
        p,q = matched(origin[sl],cv[sl],incumbent[sl],radius[sl],want,predictions['factor_prefix'][sl,:,0])
        for k,d in (('matched_independent',p),('matched_additive',q)):
            for i,path in d.items():
                paths[k][lo+i] = path
            actual = np.linalg.norm(paths[k][sl]-cv[sl],axis=-1).max(1)
            np.testing.assert_allclose(np.sort(actual[actual>0]),np.sort(want),rtol=1e-7,atol=1e-14)
    return paths


def labels(cv, y, valid, inc, cut, scale):
    err = np.linalg.norm(cv-y,axis=-1)
    count = valid.sum(1)
    known = count>0
    ref = np.nansum(err,axis=1)/count.clip(1)
    easy = known & (ref>0) & (ref<=cut)
    delta = inc-cv
    step = np.linalg.norm(delta,axis=-1)
    radius = step.max(1)
    bank = cv[:,None]+LEVELS[None,:,None,None]*delta[:,None]
    errors = np.linalg.norm(bank-y[:,None],axis=-1)
    gains = np.where(valid[:,None],err[:,None]-errors,0.).sum(2)/12.
    harms = np.maximum(-gains*12./count[:,None].clip(1),0.)
    unknown = (~known).astype(float)
    missing = np.divide((step*~valid).sum(1),step.sum(1),out=np.zeros(len(step)),where=step.sum(1)>0)
    debit = radius[:,None]*LEVELS*unknown[:,None]
    lower = gains-step.mean(1)[:,None]*LEVELS*missing[:,None]
    masses = np.column_stack((ref,np.where(easy,ref,0.)))/scale[:,None]
    direct = np.stack((lower,harms+debit,np.where(easy[:,None],harms,0.)+debit),axis=2)/scale[:,None,None]
    factor = np.stack((gains,harms,np.where(easy[:,None],harms,0.)),axis=2)/scale[:,None,None]
    return (np.column_stack((direct.reshape(len(cv),-1),masses)),
            np.column_stack((factor.reshape(len(cv),-1),masses,unknown,missing)))


def reconstruct(output, cv, inc, scale, factorized):
    raw = output[:,:9].reshape(-1,len(LEVELS),3)*scale[:,None,None]
    mass = np.broadcast_to((output[:,9:11]*scale[:,None])[:,None],(*raw.shape[:2],2))
    if factorized:
        steps = np.linalg.norm(inc-cv,axis=-1)
        unknown, missing = output[:,11:13].clip(0.,1.).T
        debit = (steps.max(1)*unknown)[:,None]*LEVELS
        raw[:,:,0] -= (steps.mean(1)*missing)[:,None]*LEVELS
        raw[:,:,1:] += debit[:,:,None]
    return np.concatenate((raw,mass),axis=2)
