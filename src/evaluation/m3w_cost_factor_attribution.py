"""Label-assisted error attribution, never an inference or deployment rule."""
import numpy as np


def _gain(new, old):
    return 100*(old-new)/old if old > 0 else None


def _parts(a, b, error, original, w):
    membership, severity, cross = [float(w@v) for v in (a*a, b*b, 2*a*b)]
    mse, old = float(w@(error*error)), float(w@(original*original))
    np.testing.assert_allclose(membership+severity+cross, mse, rtol=1e-10, atol=1e-10)
    return dict(membership_squared=membership, severity_squared=severity,
                cross_term=cross, MSE=mse, original_MSE=old,
                gain_vs_original_percent=_gain(mse, old),
                label_assisted_E_MSE=severity,
                label_assisted_E_gain_vs_original_percent=_gain(severity, old),
                label_assisted_E_gain_vs_composed_percent=_gain(severity, mse))


def diagnose(probability, easy_expert, other_expert, envelope, cv, harm, cut,
             original_easy, original_all, severity_edge, weights=None):
    arrays=[np.asarray(v, float) for v in (probability,easy_expert,other_expert,
            envelope,cv,harm,original_easy,original_all)]
    p,m,n,env,cv,h,old_e,old_a=arrays; size=len(p)
    if any(v.shape!=(size,) for v in arrays) or size==0:
        raise ValueError('Nonempty aligned vectors required')
    known=np.isfinite(cv)&np.isfinite(h)
    if (not np.array_equal(np.isnan(cv),np.isnan(h)) or
        not np.array_equal(known,~np.isnan(cv)) or
        not all(np.isfinite(v).all() for v in (p,m,n,env,old_e,old_a)) or
        not np.isfinite(cut) or cut<=0 or not np.isfinite(severity_edge) or severity_edge<0 or
        (p<0).any() or (p>1).any() or (env<0).any() or
        any((v<0).any() for v in (m,n,old_e,old_a)) or
        any((v>env+1e-4).any() for v in (m,n)) or
        (h[known]<0).any() or (h[known]>env[known]+1e-4).any()):
        raise ValueError('Finite bounded predictions and aligned known labels required')
    w=np.ones(size) if weights is None else np.asarray(weights,float)
    if w.shape!=(size,) or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Finite nonnegative weights required')
    out={}
    for name,subset in (('all',np.ones(size,bool)),('disagreement',env>0)):
        use=known&subset&(w>0)
        if not use.any():
            out[name]=dict(status='not_estimable',reason='no_known_positive_weight_support'); continue
        q=w[use]/w[use].sum(); pp,mm,nn,hh=[v[use] for v in (p,m,n,h)]
        e=((cv[use]>0)&(cv[use]<=cut)).astype(float)
        easy_error=pp*mm-e*hh
        # These two exact decompositions retain signed cross terms.
        a=(pp-e)*mm; b=e*(mm-hh)
        easy=_parts(a,b,easy_error,old_e[use]-e*hh,q)
        easy['label_assisted_H_MSE']=float(q@(((pp-e)*hh)**2))
        easy['label_assisted_H_gain_vs_original_percent']=_gain(easy['label_assisted_H_MSE'],easy['original_MSE'])
        all_error=pp*mm+(1-pp)*nn-hh
        all_parts=_parts((pp-e)*(mm-nn),e*(mm-hh)+(1-e)*(nn-hh),all_error,old_a[use]-hh,q)
        norm=float(q@(mm*mm)); mass=easy['MSE']
        masks=dict(easy=e==1,outside_easy=e==0,
                   outside_easy_low_probability=(e==0)&(pp<.1),
                   outside_easy_high_severity=(e==0)&(mm>severity_edge))
        strata={}
        for label,mask in masks.items():
            amount=float(q@mask); err=float(q[mask]@(easy_error[mask]**2))
            strata[label]=dict(rows=int(mask.sum()),weight=amount,easy_MSE_contribution=err,
                easy_MSE_share=err/mass if mass>0 else None,
                mean_probability=float(q[mask]@pp[mask]/amount) if amount>0 else None,
                mean_easy_expert=float(q[mask]@mm[mask]/amount) if amount>0 else None,
                mean_realized_harm=float(q[mask]@hh[mask]/amount) if amount>0 else None)
        out[name]=dict(rows=int(use.sum()),unknown_rows=int((~known&subset).sum()),easy=easy,all_harm=all_parts,
            membership_Brier=float(q@((pp-e)**2)),
            severity_weighted_Brier=float(q@(((pp-e)*mm)**2))/norm if norm>0 else None,
            easy_expert_squared_mean=norm,training_severity_edge=float(severity_edge),strata=strata)
    return dict(status='offline_label_assisted_diagnostic_not_deployable',subsets=out)
