"""Target-only geometric lower bound, never an inference feature or prediction."""
import numpy as np


def envelope_statistics(baseline, prediction, target, valid, budget):
    baseline,prediction,target,budget=map(lambda x:np.asarray(x,float),(baseline,prediction,target,budget))
    valid=np.asarray(valid)
    if (baseline.ndim!=3 or baseline.shape[-1]!=2 or prediction.shape!=baseline.shape
            or target.shape!=baseline.shape or valid.shape!=baseline.shape[:2]
            or valid.dtype!=bool or budget.shape!=valid.shape or (budget<0).any()
            or not all(np.isfinite(x).all() for x in (baseline,prediction,budget))
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Aligned finite predictions, causal budget and masked labels required')
    safe=np.where(valid[...,None],target,baseline)
    distance=np.linalg.norm(safe-baseline,axis=-1)
    error=np.linalg.norm(safe-prediction,axis=-1)
    used=np.linalg.norm(prediction-baseline,axis=-1)
    tolerance=1e-4+1e-5*np.maximum(budget,np.linalg.norm(baseline,axis=-1))
    if np.any(used>budget+tolerance): raise ValueError('Prediction outside declared motion envelope')
    lower=np.maximum(distance-budget,0)
    if np.any(error[valid]+tolerance[valid]<lower[valid]): raise ValueError('Invalid lower-bound computation')
    count=valid.sum(1); support=count>0
    if not support.any(): return dict(status='unsupported',rows=0)
    def ade(values): return np.where(valid,values,0).sum(1)[support]/count[support]
    ba,pa,lo=map(ade,(distance,error,lower))
    use=valid&(budget>0)
    return dict(status='defined',rows=int(support.sum()),labeled_steps=int(valid.sum()),
        baseline_ADE=float(ba.mean()),prediction_ADE=float(pa.mean()),
        oracle_envelope_lower_ADE=float(lo.mean()),
        lower_fraction_of_prediction_error=None if pa.mean()==0 else float(lo.mean()/pa.mean()),
        labels_outside_envelope_fraction=float((distance[valid]>budget[valid]+tolerance[valid]).mean()),
        zero_budget_labeled_steps=int((valid&(budget==0)).sum()),
        zero_budget_positive_error_steps=int((valid&(budget==0)&(distance>tolerance)).sum()),
        envelope_usage_mean=None if not use.any() else float((used[use]/budget[use]).mean()),
        envelope_usage_p95=None if not use.any() else float(np.quantile(used[use]/budget[use],.95)),
        fraction_above_95pct_budget=None if not use.any() else float((used[use]/budget[use]>.95).mean()))
