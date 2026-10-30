"""Past-only producer coverage and train-local detector normalization."""
import numpy as np


def coverage(sites,recordings,cached_ids,folds):
    sites,recordings,cached_ids=map(np.asarray,(sites,recordings,cached_ids))
    if (sites.ndim!=1 or recordings.shape!=sites.shape or cached_ids.ndim!=1
            or not np.issubdtype(cached_ids.dtype,np.integer) or np.any(np.diff(cached_ids)<=0)
            or (cached_ids<0).any() or (cached_ids>=len(sites)).any() or set(sites)!=set(folds)):
        raise ValueError('Exact past row identities and complete original source folds required')
    cached=np.zeros(len(sites),bool);cached[cached_ids]=True
    groups={str(s):dict(rows=int((sites==s).sum()),cached=int(((sites==s)&cached).sum()),
        records=int(len(set(recordings[sites==s])))) for s in sorted(folds)}
    jobs={}
    for kind in ('single','complement'):
        for fold in range(3):
            fit=[s for s,f in folds.items() if (f==fold)==(kind=='single')]
            mask=np.isin(sites,fit)
            if not mask.any() or mask.all():raise ValueError('Nonempty disjoint original producer roles required')
            jobs[kind+str(fold)]=dict(fit_sites=sorted(fit),train_rows=int(mask.sum()),
                missing_train_inputs=int((mask&~cached).sum()),producer_excluded_fitting_readout_rows=int((~mask&cached).sum()),
                independent_outcomes_opened=False)
    return dict(rows=len(sites),records=len(set(recordings)),cached_rows=len(cached_ids),
        missing_rows=int((~cached).sum()),cached_records=len(set(recordings[cached])),
        missing_records=len(set(recordings[~cached])),sites=groups,jobs=jobs,
        future_labels_used=False,training_run=False)


def normalize(features,sites,fit_sites):
    features,sites=np.asarray(features),np.asarray(sites)
    if (features.shape!=(len(sites),64) or not fit_sites or not set(fit_sites)<set(sites)
            or not np.isfinite(features).all()):
        raise ValueError('Full finite past features and proper own-fit source subset required')
    means=[];moments=[]
    for site in sorted(fit_sites):
        x=features[sites==site].astype(float);means.append(x.mean(0));moments.append(np.square(x).mean(0))
    mean=np.mean(means,0);std=np.sqrt(np.maximum(np.mean(moments,0)-mean**2,0)).clip(1e-6)
    return dict(mean=mean,std=std,clip=8.,fit_sites=sorted(fit_sites),future_support_used=False)
