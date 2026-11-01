"""Label-aware frozen-guard diagnosis; never an inference eligibility function."""
import numpy as np
from src.world_model.m3w_current_bank_deferral import parent


def profile(prediction, targets, allowed, scales, recordings, frames):
    p,ok=parent.allocation.base.validate(prediction,allowed);y=np.asarray(targets)
    if y.shape!=p.shape:raise ValueError('Aligned diagnostic labels required')
    known=np.isfinite(y).all((1,2))
    if not (known|np.isnan(y).all((1,2))).all():raise ValueError('Unknown labels remain wholly unknown')
    scales=np.asarray(scales);recordings=np.asarray(recordings);frames=np.asarray(frames)
    if scales.shape!=(len(p),) or (scales<=0).any():raise ValueError('Frozen positive causal scales required')
    tests=dict(positive_gain=p[:,:,0]>p[:,:,1],positive_reference=(p[:,:,2]>0)&(p[:,:,3]>0),
        all_risk=p[:,:,1]<=.02*p[:,:,2],easy_risk=p[:,:,4]<=.02*p[:,:,3])
    guard=ok.copy()
    for mask in tests.values():guard&=mask
    easy=known[:,None]&(y[:,:,3]>0)
    truth_safe=known[:,None]&ok&(y[:,:,1]<=.02*y[:,:,2])&(y[:,:,4]<=.02*y[:,:,3])
    safe_easy=truth_safe&easy
    harmful=known[:,None]&ok&((y[:,:,1]>.02*y[:,:,2])|(y[:,:,4]>.02*y[:,:,3]))
    result={}
    for rec in sorted(set(recordings)):
        at=recordings==rec;idx=np.flatnonzero(at)
        _,inverse,counts=np.unique(frames[at],return_inverse=True,return_counts=True)
        singleton=np.zeros(len(p),bool);singleton[idx]=counts[inverse]==1
        legal_easy=at[:,None]&ok&easy;kept_easy=legal_easy&guard
        rejected=at[:,None]&safe_easy&~guard
        known_easy_rows=at&known&(y[:,0,3]>0)
        n_easy=int(known_easy_rows.sum());retained=int(kept_easy.any(1).sum())
        reference=float(y[known_easy_rows,0,3].sum())
        point_block=reference>0 and retained==0
        native_joint_block=point_block and n_easy>0 and singleton[known_easy_rows].all()
        errors=(p[rejected]-y[rejected])/np.broadcast_to(scales[:,None],ok.shape)[rejected,None]
        result[str(rec)]=dict(rows=int(at.sum()),known_rows=int((at&known).sum()),unknown_rows=int((at&~known).sum()),
            full_known_easy_reference_mass=reference,known_easy_rows=n_easy,
            known_easy_rows_in_singleton_target_queries=int((known_easy_rows&singleton).sum()),
            legal_known_easy_rows=int(legal_easy.any(1).sum()),legal_known_easy_action_occurrences=int(legal_easy.sum()),
            guard_kept_known_easy_rows=retained,guard_kept_known_easy_action_occurrences=int(kept_easy.sum()),
            true_signed_risk_safe_known_easy_action_occurrences=int((at[:,None]&safe_easy).sum()),
            true_signed_risk_safe_known_easy_rejected=int(rejected.sum()),
            true_beneficial_safe_known_easy_rejected=int((rejected&(y[:,:,0]>y[:,:,1])).sum()),
            known_signed_risk_unsafe_admitted_action_occurrences=int((at[:,None]&harmful&guard).sum()),
            false_rejection_reason_occurrences={k:int((rejected&~v).sum()) for k,v in tests.items()},
            rejection_reason_counts_overlap=True,
            rejected_safe_easy_mean_normalized_cost_error=errors.mean(0).tolist() if len(errors) else None,
            frozen_pointwise_guard_cannot_supply_known_easy_reference=bool(point_block),
            frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof=bool(native_joint_block),
            original_legal_bank_has_known_easy_reference=bool(legal_easy.any()),
            diagnostic_future_labels_used=True,not_a_causal_policy_or_oracle_model_result=True)
    return result,guard
