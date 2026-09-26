"""Label-only decomposition of easy-event cost errors; never an inference rule."""
import numpy as np
from sklearn.metrics import roc_auc_score


def decompose(mean, fractional, target, cv, envelope, cut):
    mean, fractional, target, cv, envelope = map(np.asarray, (mean, fractional, target, cv, envelope))
    n=len(cv)
    if any(x.shape!=(n,4) for x in (mean,fractional,target)) or envelope.shape!=(n,) or cut<=0:
        raise ValueError('Aligned moments, causal envelope and frozen positive easy cut required')
    known=np.isfinite(target).all(1)
    if not np.array_equal(known,np.isfinite(cv)) or not np.isfinite(mean).all() or not np.isfinite(fractional).all():
        raise ValueError('Finite predictions and aligned known target support required')
    easy=(cv>0)&(cv<=cut)
    np.testing.assert_allclose(target[known,3],np.where(easy[known],target[known,1],0),rtol=0,atol=1e-10)
    use=known&(envelope>0)
    if not use.any(): return dict(status='not_estimable')
    y=target[:,3]; old=(mean[:,3]-y)**2; new=(fractional[:,3]-y)**2
    masks=dict(outside_easy_no_harm=~easy&(target[:,1]==0),
               outside_easy_positive_harm=~easy&(target[:,1]>0),
               easy_no_harm=easy&(target[:,1]==0),easy_positive_harm=easy&(target[:,1]>0))
    assert np.all(sum((v&use).astype(int) for v in masks.values())==use.astype(int))
    parts={}
    for label,mask in masks.items():
        ix=mask&use
        parts[label]=dict(rows=int(ix.sum()),mean_MSE_contribution=float(old[ix].sum()/use.sum()),
            fractional_MSE_contribution=float(new[ix].sum()/use.sum()),
            excess_MSE_contribution=float((new[ix]-old[ix]).sum()/use.sum()))
    total=float((new[use]-old[use]).mean())
    np.testing.assert_allclose(sum(v['excess_MSE_contribution'] for v in parts.values()),total,rtol=1e-10,atol=1e-10)
    rank={}
    for arm,pred in (('mean',mean),('fractional',fractional)):
        ratio=np.divide(pred[:,2],pred[:,0],out=np.zeros(n),where=pred[:,0]>0)
        rank[arm]=float(roc_auc_score(easy[use],ratio[use])) if len(set(easy[use]))==2 else None
    return dict(rows=int(use.sum()),easy=int((easy&use).sum()),excess_MSE=total,parts=parts,
        easy_reference_fraction_AUROC=rank,fraction_is_not_event_probability=True)


def summary(rows):
    out={}
    for pair in sorted({r['pair'] for r in rows}):
        ds=[f['diagnosis'] for r in rows if r['pair']==pair for f in r['folds']]
        good=[d for d in ds if d.get('status')!='not_estimable']; worse=[d for d in good if d['excess_MSE']>0]
        def contribution(d,keys): return sum(d['parts'][k]['excess_MSE_contribution'] for k in keys)
        outside=('outside_easy_no_harm','outside_easy_positive_harm')
        inside=('easy_no_harm','easy_positive_harm')
        dominant={k:sum(max(d['parts'],key=lambda k:d['parts'][k]['excess_MSE_contribution'])==k for d in worse)
                  for k in (*outside,*inside)}
        out[pair]=dict(views=len(ds),not_estimable=len(ds)-len(good),worse_views=len(worse),
            outside_easy_dominant=sum(contribution(d,outside)>contribution(d,inside) for d in worse),
            dominant_partition=dominant,
            outside_easy_share_of_positive_excess=sum(max(contribution(d,outside),0) for d in worse)/
                sum(sum(max(v['excess_MSE_contribution'],0) for v in d['parts'].values()) for d in worse) if worse else None)
    return out
