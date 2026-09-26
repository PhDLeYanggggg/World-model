"""Train-defined score bins and tie-aware harm-ranking diagnostics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def weights_for_sites(known, sites):
    known,sites=np.asarray(known,bool),np.asarray(sites)
    if known.shape!=sites.shape or not known.any(): raise ValueError('Known locality support required')
    w=np.zeros(len(sites),float)
    for site in sorted(set(sites)):
        use=known & (sites==site)
        if not use.any(): raise ValueError('Every fitting locality needs support')
        w[use]=1/(len(set(sites))*use.sum())
    return w


def event_targets(base, cv, cut):
    base=np.asarray(base,float); cv=np.asarray(cv,float)
    if base.shape!=(len(cv),4) or not np.isfinite(cut) or cut<=0:
        raise ValueError('Aligned costs and positive fitting-only cut required')
    y=base.copy(); easy=(cv>0)&(cv<=cut)
    y[:,2:]=np.where(easy[:,None],y[:,:2],0)
    y[~np.isfinite(cv)]=np.nan
    return y


def score_columns(pred, env):
    pred,env=np.asarray(pred,float),np.asarray(env,float)
    if pred.shape!=(len(env),4) or not np.isfinite(pred).all() or (env<0).any():
        raise ValueError('Aligned finite causal predictions required')
    return dict(moment=pred[:,3],fraction=np.divide(pred[:,3],env,out=np.zeros(len(env)),where=env>0),
        envelope=env)


def quantiles(score,w,probs=(.5,.9,.95,.99)):
    score,w=np.asarray(score,float),np.asarray(w,float)
    if score.shape!=w.shape or not np.isfinite(score).all() or (w<0).any() or not np.isclose(w.sum(),1):
        raise ValueError('Normalized training-only weights required')
    order=np.argsort(score,kind='stable'); c=np.cumsum(w[order])
    return np.unique(score[order[np.minimum(np.searchsorted(c,probs,side='left'),len(c)-1)]]).tolist()


def top_mass_share(score,harm,w,fraction):
    score,harm,w=map(np.asarray,(score,harm,w))
    total=float(w@harm)
    if total<=0: return None
    if not 0<fraction<=1: raise ValueError('Mass fraction in (0,1] required')
    order=np.argsort(-score,kind='stable'); s,h,p=score[order],harm[order],w[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    mass=np.add.reduceat(p,starts); cost=np.add.reduceat(p*h,starts)
    before=np.cumsum(mass)-mass
    part=np.divide(np.clip(fraction-before,0,mass),mass,out=np.zeros(len(mass)),where=mass>0)
    return float(part@cost/total)


def summarize(pred,y,env,sites,edges,subset=None):
    pred,y,env,sites=map(np.asarray,(pred,y,env,sites))
    known=np.isfinite(y).all(1)
    if subset is not None: known &= np.asarray(subset,bool)
    if not known.any(): return dict(status='not_estimable',reason='no_supported_rows')
    active=sorted(set(sites[known])); weights=np.zeros(len(y))
    for site in active:
        take=known&(sites==site); weights[take]=1/(len(active)*take.sum())
    w=weights[known]; h=y[known,3]; event=h>0; p=pred[known,3]
    positive=int(event.sum()); negative=int((~event).sum()); rate=float(w@event)
    out=dict(rows=int(known.sum()),positive=positive,negative=negative,positive_rate=rate,
        supported_localities=active,stable_event_support=positive>=20 and negative>=20,
        actual_harm_mean=float(w@h),predicted_harm_mean=float(w@p),
        harm_coverage=float((w@p)/(w@h)) if w@h>0 else None,
        harm_MSE=float(w@((p-h)**2)),
        oracle_top1_mass_share=top_mass_share(h,h,w,.01),scores={})
    for key,s in score_columns(pred,env).items():
        s=s[known]; take=np.searchsorted(np.asarray(edges[key]),s,side='left'); bins=[]
        for k in range(len(edges[key])+1):
            ix=take==k; mass=float(w[ix].sum())
            bins.append(dict(index=k,rows=int(ix.sum()),population_mass=mass,
                positive=int(event[ix].sum()),actual_harm_mass=float(w[ix]@h[ix]),
                predicted_harm_mass=float(w[ix]@p[ix])))
        au=float(roc_auc_score(event,s,sample_weight=w)) if positive and negative else None
        ap=float(average_precision_score(event,s,sample_weight=w)) if positive else None
        out['scores'][key]=dict(AUROC=au,AUPRC=ap,AP_over_prevalence=ap/rate if ap is not None else None,
            top10_harm_mass_share=top_mass_share(s,h,w,.1),bins=bins)
    return out
