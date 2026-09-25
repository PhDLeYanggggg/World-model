"""Post-fit B population diagnostics; no selection or threshold changes."""
import numpy as np


def summarize_fit(prediction, target, weights, cost_scale, rms, selected):
    pred, y, p = np.asarray(prediction,float), np.asarray(target,float), np.asarray(weights,float)
    known = np.isfinite(y).all(1)
    if pred.shape != y.shape or y.shape[1] != 4 or not np.isfinite(pred).all():
        raise ValueError('Aligned finite moments and supported targets required')
    if (p[~known]!=0).any() or not np.isclose(p.sum(),1.):
        raise ValueError('Training weights must exclude unknown labels')
    residual = np.zeros_like(y)
    residual[known] = ((pred[known]-y[known])/cost_scale/np.asarray(rms))**2
    out = {}
    for name,mask in (('population',known),('old_raw_selected',known & selected)):
        mass = p[mask].sum()
        if mass == 0:
            out[name] = dict(status='not_estimable'); continue
        w = p[mask]/mass; true = (w[:,None]*y[mask]).sum(0); fit = (w[:,None]*pred[mask]).sum(0)
        out[name] = dict(rows=int(mask.sum()), probability_mass=float(mass),
            component_mse=(w[:,None]*residual[mask]).sum(0).tolist(),
            target_moments=true.tolist(), fitted_moments=fit.tolist(),
            easy_harm_fitted_over_actual=float(fit[3]/true[3]) if true[3]>0 else None)
    return out
