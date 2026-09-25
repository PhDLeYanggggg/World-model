"""Guards and conditional moment diagnostics, not calibrated risk guarantees."""
import numpy as np


def validate_config(current, previous):
    if set(current)!=set(previous):
        raise ValueError('Same registered configuration keys required')
    for key in current:
        if key not in ('scope','arms') and current[key]!=previous[key]:
            raise ValueError('Unregistered simultaneous change: '+key)
    if current['scope']!='source_development_single_factor_symmetric_risk':
        raise ValueError('Source-only risk study required')
    if current['arms']!=['ridge','neural_mse'] or previous['arms']!=['ridge','neural_underharm4']:
        raise ValueError('Only neural risk objective changes')


def risk_diagnostic(prediction, target, selected):
    p,y,s=np.asarray(prediction),np.asarray(target),np.asarray(selected)
    if (p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=2 or s.shape!=(len(p),)
            or s.dtype!=bool or not np.isfinite(p).all() or np.any(p<0)
            or np.isinf(y).any() or not np.array_equal(np.isnan(y[:,0]),np.isnan(y[:,1]))
            or np.any(y[np.isfinite(y)]<0)):
        raise ValueError('Finite predictions and paired nonnegative supported labels required')
    known=np.isfinite(y).all(1)
    def group(mask):
        if not mask.any():
            return dict(rows=0,predicted_mean=None,true_mean=None,bias=None,mae=None,
                predicted_ratio=None,realized_ratio=None)
        a,b=p[mask].mean(0),y[mask].mean(0)
        return dict(rows=int(mask.sum()),predicted_mean=a.tolist(),true_mean=b.tolist(),
            bias=(a-b).tolist(),mae=np.abs(p[mask]-y[mask]).mean(0).tolist(),
            predicted_ratio=float(a[1]/a[0]) if a[0]>0 else None,
            realized_ratio=float(b[1]/b[0]) if b[0]>0 else None)
    return dict(all=group(known),selected=group(known&s),unknown_rows=int((~known).sum()),
        selected_unknown_rows=int((s&~known).sum()),calibrated_safety=False)
