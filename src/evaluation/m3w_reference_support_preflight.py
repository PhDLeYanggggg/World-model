"""Label-aware feasibility before training with expanded causal action support."""
import numpy as np
from src.world_model import m3w_reference_action_support as support
from src.evaluation.m3w_multiaction_support import solve_action_bank
from src.evaluation.m3w_multiaction_cost_readout import compact_score,interval

NAMES=('reference_only','reference_pointwise')
CONTROLS=('zero_joint','selected_joint')


def diagnose(packet,prediction,controls):
    p,ok=support.original.validate(prediction,packet['allowed']);y,env=packet['y'],packet['envelope']
    if set(controls)!=set(CONTROLS) or y.shape!=p.shape or env.shape!=ok.shape:raise ValueError('Original full inputs and controls required')
    masks=dict(reference_only=support.reference_actions(p,ok),reference_pointwise=support.pointwise_actions(p,ok))
    known=np.isfinite(y).all((1,2));reference=float(y[known,0,2].sum())
    if reference<=0:raise ValueError('Original positive complete SOURCE reference required')
    records={};witnesses={a:np.full(len(p),-1,np.int64) for a in NAMES}
    for rec in sorted(set(packet['recordings'])):
        at=np.flatnonzero(packet['recordings']==rec);r={}
        for a in NAMES:
            value,chosen=solve_action_bank(y[at],env[at],masks[a][at]);witnesses[a][at]=chosen;r[a]=value
        records[str(rec)]=r
    for g in support.original.query_groups(packet['recordings'],packet['frames'],packet['ids']):
        if not support.feasible(p[g],ok[g],witnesses['reference_pointwise'][g]):raise AssertionError('Original two2%query budgets must hold')
    result={}
    for a in NAMES:
        if (witnesses[a][~known]>=0).any():raise AssertionError('Unknown outcomes cannot supply oracle labels')
        score=compact_score(y,env,witnesses[a])
        result[a]=dict(allowed_action_occurrences=int(masks[a].sum()),witness=score,
            witness_gain_pp=100*score['selected_net_gain_lower_mass']/reference,
            supported_record_views=sum(r[a]['budget_feasible'] for r in records.values()),
            positive_record_views=sum(r[a]['positive_feasible'] for r in records.values()))
    return dict(rows=len(p),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),reference_mass=reference,
        supports=result,recordings=records,controls={a:compact_score(y,env,c) for a,c in controls.items()},
        all_reference_pointwise_query_budgets_verified=True,learned_moments_unchanged=True,
        only_removed_restriction='old_predicted_benefit_must_exceed_predicted_harm',
        original2percent_risk_tolerance_unchanged=True,free_counts_not_trained_actor_counts=True,
        witness_uses_future_labels=True,witness_not_model_result=True,new_training=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,sites in contexts.items() for s in sites for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:raise ValueError('All72original groups required')
    output={}
    for a in NAMES:
        records=[v[a] for r in rows for v in r['result']['recordings'].values()]
        output[a]=dict(witness_gain_pp=interval(rows,lambda r,a=a:r['supports'][a]['witness_gain_pp'],cfg),
            supported_record_views=sum(r['budget_feasible'] for r in records),positive_record_views=sum(r['positive_feasible'] for r in records),
            required_record_views=len(records),infeasible_reasons={reason:sum(r['infeasibility_reason']==reason for r in records)
                for reason in sorted({r['infeasibility_reason'] for r in records if r['infeasibility_reason']})})
    return dict(groups=72,supports=output,
        conservative_pointwise_preflight_pass=output['reference_pointwise']['positive_record_views']==output['reference_pointwise']['required_record_views'],
        pointwise_failure_is_not_joint_infeasibility_proof=True,
        all_predicted_query_budgets_exact=all(r['result']['all_reference_pointwise_query_budgets_verified'] for r in rows),
        label_aware_diagnostic_only=True,oracle_not_model_success=True,new_training=False,
        independent_roles_opened=False,independent_confirmation=False,whole_pipeline_OOF=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
