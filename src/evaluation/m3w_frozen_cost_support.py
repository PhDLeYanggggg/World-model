"""Label-aware support diagnosis only; never an inference selector or training label."""
import numpy as np
from src.evaluation.m3w_multiaction_support import solve_action_bank
from src.evaluation.m3w_multiaction_cost_readout import compact_score,interval
from src.world_model import m3w_frame_action_allocation as allocation

SUPPORTS=('causal_available','predicted_positive','pointwise_cost_guard')
CONTROLS=('zero_joint','selected_joint')


def candidate_masks(prediction,allowed):
    p,ok=allocation.validate(prediction,allowed)
    positive=allocation.positive_actions(p,ok)
    safe=positive&(p[:,:,1]<=.02*p[:,:,2])&(p[:,:,4]<=.02*p[:,:,3])
    return dict(causal_available=ok,predicted_positive=positive,pointwise_cost_guard=safe)


def upper_benefit(y,allowed):
    known=np.isfinite(y).all((1,2))
    value=np.where(allowed&known[:,None],y[:,:,0],0.)
    return float(value.max(1).sum())


def diagnose(packet,prediction,controls):
    y,env,allowed=packet['y'],packet['envelope'],packet['allowed']
    p,allowed=allocation.validate(prediction,allowed)
    if set(controls)!=set(CONTROLS):raise ValueError('Both complete frozen controls required')
    if y.shape!=p.shape or env.shape!=p.shape[:2]:raise ValueError('Original native-unit labels required')
    masks=candidate_masks(p,allowed);known=np.isfinite(y).all((1,2))
    reference=float(y[known,0,2].sum())
    if reference<=0:raise ValueError('Original complete reference required')
    witnesses={a:np.full(len(p),-1,np.int64) for a in SUPPORTS}
    records={};policies={a:compact_score(y,env,c) for a,c in controls.items()}
    for rec in sorted(set(packet['recordings'])):
        at=np.flatnonzero(packet['recordings']==rec);rr={}
        for arm in SUPPORTS:
            result,choice=solve_action_bank(y[at],env[at],masks[arm][at]);witnesses[arm][at]=choice
            rr[arm]=dict(oracle=result,positive_benefit_upper_mass=upper_benefit(y[at],masks[arm][at]))
        rr['controls']={}
        for arm,c in controls.items():
            chosen=c[at];only=np.zeros((len(at),7),bool);selected=np.flatnonzero(chosen>=0)
            only[selected,chosen[selected]]=True
            deletion,_=solve_action_bank(y[at],env[at],only)
            rr['controls'][arm]=dict(actual=compact_score(y[at],env[at],chosen),label_aware_deletion=deletion)
        records[str(rec)]=rr
    for g in allocation.query_groups(packet['recordings'],packet['frames'],packet['ids']):
        if not allocation.feasible(p[g],allowed[g],witnesses['pointwise_cost_guard'][g]):
            raise AssertionError('Pointwise-safe witness must satisfy every original predicted query budget')
    if any((c[~known]>=0).any() for c in witnesses.values()):raise AssertionError('Unknown labels cannot become oracle selections')
    output={}
    for arm,mask in masks.items():
        score=compact_score(y,env,witnesses[arm])
        output[arm]=dict(allowed_action_occurrences=int(mask.sum()),known_allowed_action_occurrences=int((mask&known[:,None]).sum()),
            known_positive_rows=int((mask&known[:,None]&(y[:,:,0]>0)).any(1).sum()),
            positive_benefit_upper_mass=upper_benefit(y,mask),positive_benefit_upper_pp=100*upper_benefit(y,mask)/reference,
            witness=score,witness_gain_pp=100*score['selected_net_gain_lower_mass']/reference,
            witness_supported_records=sum(r[arm]['oracle']['budget_feasible'] for r in records.values()),
            witness_positive_records=sum(r[arm]['oracle']['positive_feasible'] for r in records.values()),
            unsupported_record_reasons={rec:r[arm]['oracle']['infeasibility_reason'] for rec,r in records.items()
                if not r[arm]['oracle']['budget_feasible']},
            label_aware_not_deployable=True,free_count_not_original_actor_count=True)
    calibration={}
    for arm,c in controls.items():
        at=np.flatnonzero((c>=0)&known);a=c[at];q=p[at,a];truth=y[at,a]
        frozen_safe=(q[:,1]<=.02*q[:,2])&(q[:,4]<=.02*q[:,3])
        observed_bad=(truth[:,1]>.02*truth[:,2])|(truth[:,4]>.02*truth[:,3])
        calibration[arm]=dict(known_selected_rows=len(at),predicted_pointwise_safe_rows=int(frozen_safe.sum()),
            predicted_pointwise_safe_but_observed_row_risk_over2_percent=int((frozen_safe&observed_bad).sum()),
            predicted_summed_moments=q.sum(0).tolist(),observed_summed_moments=truth.sum(0).tolist(),
            row_diagnostic_not_recording_safety=True)
    return dict(rows=len(p),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),reference_mass=reference,
        supports=output,controls=policies,recordings=records,calibration=calibration,
        pointwise_witness_all_original_query_budgets_verified=True,
        predicted_positive_is_not_query_feasibility=True,pointwise_guard_is_restriction_not_joint_feasible_set=True,
        causal_available_oracle_is_upper_control_not_inference=True,
        labels_used_by_oracle=True,unknown_rows_never_oracle_selected=True,new_training=False,
        independent_roles_opened=False,deployment_changed=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,sites in contexts.items() for s in sites for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:
        raise ValueError('Complete72 original SOURCE groups required')
    result={}
    for arm in SUPPORTS:
        records=[v[arm]['oracle'] for r in rows for v in r['result']['recordings'].values()]
        result[arm]=dict(positive_upper_pp=interval(rows,lambda r,a=arm:r['supports'][a]['positive_benefit_upper_pp'],cfg),
            safe_witness_gain_pp=interval(rows,lambda r,a=arm:r['supports'][a]['witness_gain_pp'],cfg),
            supported_record_views=sum(r['budget_feasible'] for r in records),positive_record_views=sum(r['positive_feasible'] for r in records),
            required_record_views=len(records),infeasible_reasons={reason:sum(r['infeasibility_reason']==reason for r in records)
                for reason in sorted({r['infeasibility_reason'] for r in records if r['infeasibility_reason']})})
    losses={}
    for left,right in zip(SUPPORTS,SUPPORTS[1:]):
        losses[left+'_minus_'+right]=interval(rows,lambda r,l=left,b=right:
            r['supports'][l]['positive_benefit_upper_pp']-r['supports'][b]['positive_benefit_upper_pp'],cfg)
    controls={}
    for arm in CONTROLS:
        records=[r['controls'][arm] for row in rows for r in row['result']['recordings'].values()]
        cal=[r['result']['calibration'][arm] for r in rows]
        controls[arm]=dict(actual_supported_record_views=sum(r['actual']['budget_status']['supported'] for r in records),
            label_aware_deletion_supported_record_views=sum(r['label_aware_deletion']['budget_feasible'] for r in records),
            label_aware_deletion_positive_record_views=sum(r['label_aware_deletion']['positive_feasible'] for r in records),
            known_selected_row_views=sum(r['known_selected_rows'] for r in cal),
            predicted_pointwise_safe_row_views=sum(r['predicted_pointwise_safe_rows'] for r in cal),
            predicted_pointwise_safe_but_observed_row_risk_over2_percent=sum(r['predicted_pointwise_safe_but_observed_row_risk_over2_percent'] for r in cal))
    return dict(groups=72,supports=result,upper_benefit_losses=losses,controls=controls,
        full_cohort_preserved=True,pointwise_query_budget_witnesses_exact=all(r['result']['pointwise_witness_all_original_query_budgets_verified'] for r in rows),
        label_aware_diagnostic_only=True,oracle_not_a_model_result=True,oracle_not_an_inference_input=True,
        free_count_witness_not_matched_count_upper_bound=True,whole_pipeline_OOF=False,independent_confirmation=False,
        new_training=False,held_scored=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
