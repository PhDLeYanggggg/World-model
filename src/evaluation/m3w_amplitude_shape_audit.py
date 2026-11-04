"""Fitting-only cross-amplitude diagnostics, not an inference or safety rule."""
import numpy as np

ALPHA=np.array([0.,1/32,1/16,1/8,1/4,1/2,1.])
TOLERANCE=1e-6


def flags(value,active):
    slopes=np.diff(value,axis=1)/np.diff(ALPHA)
    ratio=value[:,1:]/ALPHA[1:]
    return dict(nonmonotone_rows=int((active & (np.diff(value,axis=1)<-TOLERANCE).any(1)).sum()),
        nonconvex_rows=int((active & (np.diff(slopes,axis=1)<-TOLERANCE).any(1)).sum()),
        ratio_order_violating_rows=int((active & (np.diff(ratio,axis=1)<-TOLERANCE).any(1)).sum()))


def diagnose(predicted,truth,known,choice,envelope):
    p,y,k,c,e=map(np.asarray,(predicted,truth,known,choice,envelope))
    n=len(p)
    if (p.shape!=(n,7,5)or y.shape!=p.shape or e.shape!=(n,7)or k.shape!=(n,)
            or k.dtype!=bool or c.shape!=(n,)or c.dtype.kind not in 'iu'
            or np.any((c< -1)|(c>6)|(c==0))
            or any(not np.isfinite(v).all()or(v<0).any()for v in (p,y,e))):
        raise ValueError('Finite aligned amplitude moments, separate labels and non-alias choices required')
    active=e[:,6]>0
    selected=c>0
    ix=np.flatnonzero(selected & k)
    selected_EH=float(y[ix,c[ix],4].sum())
    result=dict(rows=n,active_rows=int(active.sum()),selected=int(selected.sum()),
        selected_known=int((selected & k).sum()),selected_known_EH_mass=selected_EH,
        predicted={},bounded_labels={},selected_ratio_inversion={})
    for name,j in (('H',1),('EH',4)):
        norm=lambda v:np.divide(v[:,:,j],e[:,6,None],out=np.zeros((n,7)),where=active[:,None])
        a,b=norm(p),norm(y)
        result['predicted'][name]=flags(a,active)
        result['bounded_labels'][name]=flags(b,active)
        ratios=a[:,1:]/ALPHA[1:]
        earlier=np.concatenate((np.zeros((n,1)),np.maximum.accumulate(ratios,axis=1)[:,:-1]),1)
        row=np.arange(n);slot=np.maximum(c-1,0)
        bad=selected & active & (ratios[row,slot]+TOLERANCE<earlier[row,slot])
        at=np.flatnonzero(bad & k)
        mass=float(y[at,c[at],4].sum())
        result['selected_ratio_inversion'][name]=dict(selected_rows=int(bad.sum()),
            known_EH_mass=mass,known_EH_mass_share=mass/selected_EH if selected_EH>0 else None)
    spans=[]
    for j in (2,3):
        scale=p[:,:,j].max(1)
        relative=np.divide(np.ptp(p[:,:,j],axis=1),scale,out=np.zeros(n),where=scale>0)
        spans.append(float(relative.max(initial=0.)))
    result['reference_max_relative_span']=spans
    result['normalized_curve_tolerance']=TOLERANCE
    result['labels_used_for_selection']=False
    return result
