"""Original-risk evaluation of coherent marginal and identified cost factors."""
import numpy as np
from src.evaluation.m3w_own_decision_cost import old
from src.evaluation.m3w_current_guard_support import profile
from src.world_model.m3w_frame_action_allocation import query_groups

NAMES=('marginal_guarded','identified_guarded','frozen_moment_guarded','frozen_boundary_guarded',
       'frozen_tdef_guarded','frozen_own_joint','tdef_raw','floor')
PAIRS=(('identified_minus_marginal','marginal_guarded','identified_guarded'),
       ('identified_minus_frozen_guard','frozen_tdef_guarded','identified_guarded'),
       ('identified_minus_frozen_joint','frozen_own_joint','identified_guarded'))


def readout(packet,choices,ranks,predictions,probabilities,easy_cut,origins):
    choices=dict(choices,floor=np.full(len(packet['ids']),-1,np.int64))
    if set(choices)!=set(NAMES):raise ValueError('Both factors and complete frozen controls required')
    policies={a:old.bank.policy_metrics(packet,c) for a,c in choices.items()}
    reference=float(packet['floor_ade'][np.isfinite(packet['floor_ade'])].sum())
    records={str(r):{a:old.bank.legacy.compact_score(packet['y'][at],packet['envelope'][at],c[at]) for a,c in choices.items()}
        for r in sorted(set(packet['recordings'])) for at in (packet['recordings']==r,)}
    comparisons={n:old.bank.pair(packet,choices[a],ranks[a],packet,choices[b],ranks[b]) for n,a,b in PAIRS}
    ade={a:old.bank.legacy.chosen_values(packet['ade'],packet['floor_ade'],c) for a,c in choices.items()}
    known=np.isfinite(packet['y']).all((1,2));legal=known[:,None]&packet['allowed']
    event=(packet['cv_ade']>0)&(packet['cv_ade']<=easy_cut)
    diagnostics={};moments={};event_score={}
    for a,p in predictions.items():
        diagnostics[a]=profile(p,packet['y'],packet['allowed'],packet['scales'],packet['recordings'],packet['frames'])[0]
        error=(p-packet['y'])/packet['scales'][:,None,None]
        moments[a]=np.square(error[legal]).mean(0).tolist() if legal.any() else None
        event_score[a]=float(np.square(probabilities[a][known]-event[known]).mean()) if known.any() else None
    return dict(policies=policies,recordings=records,comparisons=comparisons,
        floor_lower={a:100*p['policy']['selected_net_gain_lower_mass']/reference for a,p in policies.items()},
        subsets=old.subsets(packet,choices,easy_cut),
        pair_proxy=old.pair_proxy(packet,query_groups(packet['recordings'],packet['frames'],packet['ids']),choices,origins),
        record_tails={str(r):{a:old.tail_stats(v,packet['recordings']==r) for a,v in ade.items()} for r in records},
        guard_diagnostic=diagnostics,normalized_moment_MSE=moments,easy_occurrence_Brier=event_score,
        inference_uses_outcomes=False,diagnostics_only_after_frozen_choices=True,new_trajectory_training=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,ss in contexts.items() for s in ss for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:
        raise ValueError('All original 72 SOURCE groups required')
    interval=lambda fn:old.bank.legacy.interval(rows,fn,cfg)
    effects={n:{m:{k:interval(lambda r,n=n,m=m,k=k:r['comparisons'][n][m][k]) for k in ('known_pp','lower_pp','upper_pp')} for m in ('full','matched')} for n,_,_ in PAIRS}
    safety={a:old.bank.safety(rows,a) for a in NAMES}
    floor={a:interval(lambda r,a=a:r['floor_lower'][a]) for a in NAMES}
    positive=lambda x:x['CI95'] is not None and x['CI95'][0]>0
    risk=safety['identified_guarded']
    gates=dict(complete36fits72groups=True,matched_capacity_initial_draws_budget=all(r['matched_training'] for r in rows),
        complete_query_coverage=all(r['minimum_gradient_query_visits']>=4 for r in rows),
        original_controls_exact=all(r['original_control_exact'] for r in rows),
        coherent_predictions_and_original_guard=all(r['guarded_predicted_feasible'] for r in rows),
        primary_identified_minus_marginal_full_lower_CI_positive=positive(effects['identified_minus_marginal']['full']['lower_pp']),
        identified_minus_frozen_full_lower_CI_positive=positive(effects['identified_minus_frozen_guard']['full']['lower_pp']),
        identified_floor_lower_CI_positive=positive(floor['identified_guarded']),
        all_original_recording_risk=risk['supported_views']==72 and risk['supported_record_views']==risk['required_record_views'])
    diagnostic={}
    for a in ('marginal','identified'):
        cells=[v for r in rows for v in r['result']['guard_diagnostic'][a].values()]
        diagnostic[a]=dict(repeated_record_views=len(cells),
            pointwise_blocked_views=sum(v['frozen_pointwise_guard_cannot_supply_known_easy_reference'] for v in cells),
            native_joint_singleton_proof_views=sum(v['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof'] for v in cells),
            safe_easy_rejected=sum(v['true_signed_risk_safe_known_easy_rejected'] for v in cells),
            beneficial_safe_easy_rejected=sum(v['true_beneficial_safe_known_easy_rejected'] for v in cells),
            known_unsafe_admitted=sum(v['known_signed_risk_unsafe_admitted_action_occurrences'] for v in cells))
    return dict(groups=72,new_fits=36,primary='identified_minus_marginal_full_completion_lower',
        effects=effects,safety=safety,floor_lower=floor,guard_diagnostic=diagnostic,
        accuracy={a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a:interval(lambda r,a=a:r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        normalized_moment_MSE={a:{str(k):interval(lambda r,a=a,k=k:None if r['normalized_moment_MSE'][a] is None else r['normalized_moment_MSE'][a][k]) for k in range(5)} for a in ('marginal','identified')},
        easy_occurrence_Brier={a:interval(lambda r,a=a:r['easy_occurrence_Brier'][a]) for a in ('marginal','identified')},
        subsets=old.pooled_subsets(rows),fitting_gates=gates,fitting_advancement=all(gates.values()),
        source_role='exposed_SOURCE_fitting_resubstitution_only',independent_confirmation=False,
        whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,
        new_trajectory_training=False,stage5c_executed=False,smc_enabled=False)
