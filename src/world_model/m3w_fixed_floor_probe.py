"""Fixed-producer incremental cost probes; no future fields in prediction APIs."""
import numpy as np
import torch


def assert_roles(producer, controller, fitting, held):
    groups = list(map(set, (producer, controller, fitting, held)))
    if [len(s) for s in groups] != [4,4,2,2] or len(set.union(*groups)) != 12:
        raise ValueError('Disjoint four/four/two/two source roles required')


def targets(cv, reference, candidate, easy_cut):
    cv, r, n = [np.asarray(a, float) for a in (cv, reference, candidate)]
    known = np.isfinite(cv)
    if (cv.ndim != 1 or r.shape != cv.shape or n.shape != cv.shape or
            not np.array_equal(known,np.isfinite(r)) or not np.array_equal(known,np.isfinite(n)) or
            any(np.isinf(a).any() or (a[known]<0).any() for a in (cv,r,n)) or easy_cut<=0):
        raise ValueError('Aligned nonnegative costs with identical unknown support required')
    easy = known & (cv>0) & (cv<=easy_cut)
    benefit, harm = np.maximum(r-n,0), np.maximum(n-r,0)
    y = np.column_stack((benefit,harm,r,np.where(easy,r,0),np.where(easy,harm,0)))
    y[~known] = np.nan
    return y


def fit_ridge(x, y, cv, sites, *, ridge_lambda, clip):
    x,y,cv,sites = np.asarray(x,float),np.asarray(y,float),np.asarray(cv,float),np.asarray(sites)
    known=np.isfinite(cv); fit_sites=sorted(set(sites))
    if (x.ndim!=2 or y.shape[0]!=len(x) or cv.shape!=(len(x),) or len(sites)!=len(x) or
            not np.isfinite(x).all() or not np.array_equal(np.isfinite(y).all(1),known) or
            np.isinf(y).any() or (y[known]<0).any() or ridge_lambda<=0 or clip<=0 or
            any(not np.any(known & (sites==s)) for s in fit_sites)):
        raise ValueError('Fitting-only finite features and aligned known labels required')
    w=np.zeros(len(x),float)
    for s in fit_sites:
        ix=known & (sites==s); w[ix]=1/(len(fit_sites)*ix.sum())
    mean=(x*w[:,None]).sum(0)
    std=np.sqrt(((x-mean)**2*w[:,None]).sum(0)); std=np.maximum(std,1e-6)
    scale=float(np.dot(w[known],cv[known]))
    if not np.isfinite(scale) or scale<=0: raise ValueError('Positive training-only cost scale required')
    constant=(y[known]/scale*w[known,None]).sum(0)
    gram=torch.zeros((x.shape[1],x.shape[1]),dtype=torch.float64)
    rhs=torch.zeros((x.shape[1],y.shape[1]),dtype=torch.float64)
    ids=np.flatnonzero(known)
    for first in range(0,len(ids),4096):
        ix=ids[first:first+4096]
        z=torch.from_numpy(np.clip((x[ix]-mean)/std,-clip,clip))
        wt=torch.from_numpy(w[ix,None])
        yt=torch.from_numpy(y[ix]/scale-constant)
        gram+=z.T@(z*wt); rhs+=z.T@(yt*wt)
    gram+=ridge_lambda*torch.eye(x.shape[1],dtype=torch.float64)
    coef=torch.linalg.solve(gram,rhs).numpy()
    return dict(mean=mean,std=std,scale=scale,constant=constant,coef=coef,
        ridge_lambda=ridge_lambda,clip=clip,training_sites=fit_sites,known_rows=int(known.sum()),
        unknown_rows=int((~known).sum()),site_weight={s:float(w[sites==s].sum()) for s in fit_sites})


def predict(model,x,envelope):
    x,envelope=np.asarray(x,float),np.asarray(envelope,float)
    if (x.ndim!=2 or x.shape[1]!=len(model['mean']) or envelope.shape!=(len(x),) or
            not np.isfinite(x).all() or not np.isfinite(envelope).all() or (envelope<0).any()):
        raise ValueError('Only finite causal features and rollout envelope accepted')
    z=(x-model['mean'])/model['std']; distance=np.sqrt(np.mean(z*z,axis=1))
    coef=torch.from_numpy(model['coef']); outputs=[]
    for first in range(0,len(x),4096):
        xt=torch.from_numpy(np.clip(z[first:first+4096],-model['clip'],model['clip']))
        outputs.append((xt@coef).numpy())
    out=np.maximum((np.concatenate(outputs)+model['constant'])*model['scale'],0)
    for i in range(0,out.shape[1],5):
        mass=out[:,i:i+2].sum(1)
        out[:,i:i+2]*=np.minimum(1,np.divide(envelope,mass,out=np.ones(len(mass)),where=mass>0))[:,None]
        out[:,i+4]=np.minimum(out[:,i+4],envelope)
    return out,distance


def decisions(p,moving,supported):
    p,moving,supported=np.asarray(p),np.asarray(moving),np.asarray(supported)
    if (p.shape!=(len(moving),5) or moving.dtype!=bool or supported.dtype!=bool or
            supported.shape!=moving.shape or not np.isfinite(p).all() or (p<0).any()):
        raise ValueError('Five nonnegative predicted moments and explicit causal guards required')
    positive=moving & (p[:,0]>p[:,1])
    return positive, positive & supported & (p[:,1]<=.02*p[:,2]) & (p[:,4]<=.02*p[:,3])


def accounting(cv,floor,neural,take):
    cv,floor,neural,take=map(np.asarray,(cv,floor,neural,take))
    known=np.isfinite(cv)
    if take.dtype!=bool or take.shape!=cv.shape: raise ValueError('Aligned Boolean action required')
    targets(cv,floor,neural,1.)
    b,h=np.maximum(floor-neural,0),np.maximum(neural-floor,0)
    yes=known & take; no=known & ~take
    out=dict(captured_benefit=float(b[yes].sum()),selected_harm=float(h[yes].sum()),
        missed_benefit=float(b[no].sum()),oracle_benefit=float(b[known].sum()),
        fallback_regression=float(np.maximum(cv-floor,0)[no].sum()),
        fallback_relief=float(np.maximum(floor-cv,0)[no].sum()),unknown_rows=int((~known).sum()))
    out['rebased_gain_sum']=float((floor[known]-np.where(take,neural,floor)[known]).sum())
    out['original_gain_sum']=float((floor[known]-np.where(take,neural,cv)[known]).sum())
    np.testing.assert_allclose(out['rebased_gain_sum'],out['captured_benefit']-out['selected_harm'],atol=1e-8)
    np.testing.assert_allclose(out['original_gain_sum'],out['rebased_gain_sum']-out['fallback_regression']+out['fallback_relief'],atol=1e-8)
    return out
