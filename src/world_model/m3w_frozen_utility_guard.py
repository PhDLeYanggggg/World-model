"""Recombine frozen predicted utility and risk without labels or new thresholds."""
import numpy as np
from src.world_model.m3w_multiaction_cost import select_actions

POLICIES=("coupled_disabled","coupled_real","coupled_noise","fixed_utility_real_guard","fixed_utility_noise_guard")


def select(utility_prediction, risk_prediction, allowed):
    p,r,a=np.asarray(utility_prediction),np.asarray(risk_prediction),np.asarray(allowed)
    if (p.ndim!=3 or p.shape[1:]!=(7,5) or r.shape!=p.shape or a.shape!=p.shape[:2]
            or a.dtype!=bool or not np.isfinite(p).all() or not np.isfinite(r).all()
            or (p<0).any() or (r<0).any()):raise ValueError("Aligned frozen predicted costs and causal mask required")
    gain=p[...,0]-p[...,1]
    legal=a&(gain>0)&(r[...,2]>0)&(r[...,3]>0)&(r[...,1]<=.02*r[...,2])&(r[...,4]<=.02*r[...,3])
    best=np.where(legal,gain,-np.inf).argmax(1)
    return np.where(legal.any(1),best,-1).astype(np.int64)


def policies(predictions, allowed, saved):
    if set(predictions)!={"disabled","real","noise"} or set(saved)!=set(predictions):
        raise ValueError("All three frozen arms required")
    out={}
    for arm in predictions:
        c,_=select_actions(predictions[arm],allowed,arm="physical_neural")
        np.testing.assert_array_equal(c,saved[arm]);out["coupled_"+arm]=c
    for arm in ("real","noise"):
        out["fixed_utility_"+arm+"_guard"]=select(predictions["disabled"],predictions[arm],allowed)
    return {k:out[k] for k in POLICIES}


def causal_diagnosis(predictions,allowed,choices):
    p,r=predictions["disabled"],predictions["real"]
    gain=p[...,0]-p[...,1];other=r[...,0]-r[...,1]
    guard=allowed&(r[...,2]>0)&(r[...,3]>0)&(r[...,1]<=.02*r[...,2])&(r[...,4]<=.02*r[...,3])
    positive=guard&(gain>0)
    return dict(row_views=len(p),original_positive_real_guard_action_views=int(positive.sum()),
        original_positive_real_guard_but_new_gain_nonpositive_action_views=int((positive&(other<=0)).sum()),
        switch_counts={k:int((v>=0).sum()) for k,v in choices.items()},future_labels_used=False,
        interpretation="predicted_opportunities_only_not_proved_useful_or_safe")
