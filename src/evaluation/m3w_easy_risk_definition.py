"""Post-decision reliability accounting; never an inference feature builder."""
import numpy as np


def sums(cv, candidate, complete, selected, q_hat, r_hat, cutoff):
    b, e, k, x, q, r = map(np.asarray, (cv, candidate, complete, selected, q_hat, r_hat))
    if (any(v.shape!=b.shape for v in (e,k,x,q,r)) or b.ndim!=1
            or k.dtype!=bool or x.dtype!=bool or cutoff<=0 or not np.isfinite(cutoff)
            or not np.isfinite(b[k]).all() or not np.isfinite(e[k]).all()
            or not np.isfinite(q).all() or not np.isfinite(r).all()
            or (q<0).any() or (r<0).any()):
        raise ValueError('Aligned post-decision costs and predicted nonnegative moments required')
    use=k&x; easy=k&(b<=cutoff)
    harm=np.where(easy,np.maximum(e-b,0),0)
    benefit=np.where(easy,np.maximum(b-e,0),0)
    denominator=np.where(easy,b,0)
    h=float(harm[use].sum());g=float(benefit[use].sum())
    den=float(denominator[use].sum());pop=float(denominator[k].sum())
    ph=float(q[use].sum());pd=float(r[use].sum())
    def ratio(a,b):return 100*a/b if b>0 else None
    return dict(selected=int(x.sum()),complete_selected=int(use.sum()),
        incomplete_selected=int((x&~k).sum()),easy_selected=int((easy&use).sum()),
        zero_CV_harmed=int((use&(b==0)&(e>0)).sum()),
        predicted_harm=ph,observed_positive_harm=h,observed_easy_benefit=g,
        observed_net_easy_harm=h-g,predicted_selected_denominator=pd,
        observed_selected_denominator=den,observed_population_denominator=pop,
        predicted_positive_selected_ratio=ratio(ph,pd),
        observed_positive_selected_ratio=ratio(h,den),
        observed_net_selected_ratio=ratio(h-g,den),
        observed_positive_population_ratio=ratio(h,pop),
        observed_net_population_ratio=ratio(h-g,pop),
        benefit_cancels_positive_harm_percent=ratio(g,h),
        predicted_to_observed_harm=ph/h if h>0 else None,
        predicted_to_observed_denominator=pd/den if den>0 else None)
