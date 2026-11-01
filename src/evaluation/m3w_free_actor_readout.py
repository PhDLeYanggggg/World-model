"""Complete original SOURCE scoring for separate utility and cost heads."""
import numpy as np
from src.evaluation import m3w_coupled_benefit_risk as old
from src.world_model import m3w_actor_allocation as decoder

NAMES=(*decoder.NAMES,'decision_v1_joint','floor')
COMPARISONS=dict(selected_minus_zero=('zero','selected'),self_minus_zero=('zero','self'),selected_minus_self=('self','selected'))


def readout(packet,choices,ranks,info,easy_cut,origins):
    choices=dict(choices,floor=np.full(len(packet['ids']),-1,np.int64))
    policies={a:old.bank.policy_metrics(packet,c) for a,c in choices.items()}
    recordings={str(rec):{a:old.bank.legacy.compact_score(packet['y'][at],packet['envelope'][at],c[at]) for a,c in choices.items()}
        for rec in sorted(set(packet['recordings'])) for at in (packet['recordings']==rec,)}
    comparisons={name:{mode:old.bank.pair(packet,choices[a+'_'+mode],ranks[a+'_'+mode],packet,choices[b+'_'+mode],ranks[b+'_'+mode])
        for mode in ('joint','matched')} for name,(a,b) in COMPARISONS.items()}
    joint_effects={a:old.bank.pair(packet,choices[a+'_independent'],ranks[a+'_independent'],packet,choices[a+'_joint'],ranks[a+'_joint']) for a in ('zero','self','selected')}
    legacy_effect=old.bank.cross_bank_bounds(packet,choices['decision_v1_joint'],packet,choices['selected_joint'])
    reference=packet['floor_ade'][np.isfinite(packet['floor_ade'])].sum()
    ade={a:old.bank.legacy.chosen_values(packet['ade'],packet['floor_ade'],c) for a,c in choices.items()}
    return dict(policies=policies,recordings=recordings,comparisons=comparisons,joint_effects=joint_effects,selected_minus_legacy=legacy_effect,
        floor_lower={a:100*v['policy']['selected_net_gain_lower_mass']/reference if reference>0 else None for a,v in policies.items()},
        subsets=old.subsets(packet,choices,easy_cut),allocation=info,
        pair_proxy=old.pair_proxy(packet,decoder.base.query_groups(packet['recordings'],packet['frames'],packet['ids']),choices,origins),
        record_tails={str(rec):{a:old.tail_stats(e,packet['recordings']==rec) for a,e in ade.items()} for rec in recordings},
        original_targets_references_and_all_required_rows_unchanged=True,inference_outcomes_used=False,
        whole_pipeline_OOF=False,new_trajectory_training=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,seed) for c,ss in contexts.items() for s in ss for seed in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:raise ValueError('Complete72 source views required')
    interval=lambda get:old.bank.legacy.interval(rows,get,cfg)
    contrasts={name:{mode:{bound:interval(lambda r,n=name,m=mode,b=bound:r['comparisons'][n][m]['full'][b])
        for bound in ('known_pp','lower_pp','upper_pp')} for mode in ('joint','matched')} for name in COMPARISONS}
    joint={a:{bound:interval(lambda r,a=a,b=bound:r['joint_effects'][a]['full'][b]) for bound in ('known_pp','lower_pp','upper_pp')}
        for a in ('zero','self','selected')}
    safety={a:old.bank.safety(rows,a) for a in NAMES}
    lower={a:interval(lambda r,a=a:r['floor_lower'][a]) for a in NAMES}
    positive=lambda v:v['CI95'] is not None and v['CI95'][0]>0
    risk=safety['selected_joint']
    gates=dict(complete36actors72groups=True,all72_zero_and_legacy_full_scores_exact=all(r['old_controls_exact'] for r in rows),
        matched_context_arms=all(r['matched_training_verified'] for r in rows),cost_moments_frozen=all(r['cost_moments_exact'] for r in rows),
        same_count_controls_every_query=all(r['result']['allocation']['same_count_controls_every_query'] for r in rows),
        primary_selected_minus_zero_lower_CI_positive=positive(contrasts['selected_minus_zero']['joint']['lower_pp']),
        own_floor_lower_CI_positive=positive(lower['selected_joint']),
        complete_original_recording_risk=risk['supported_views']==72 and risk['supported_record_views']==risk['required_record_views'])
    return dict(groups=72,new_actor_fits=36,primary_optimizer_updates=72000,exact_refit_updates=4000,
        primary='selected_context_joint_minus_zero_gate_joint_full_completion_lower',contrasts=contrasts,joint_effects=joint,
        selected_minus_legacy={bound:interval(lambda r,b=bound:r['selected_minus_legacy'][b]) for bound in ('known_pp','lower_pp','upper_pp')},
        accuracy={a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a:interval(lambda r,a=a:r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        safety=safety,floor_lower=lower,subsets=old.pooled_subsets(rows),fitting_gates=gates,fitting_advancement_gate=all(gates.values()),
        component_checks=dict(context_matched_lower_CI_positive=positive(contrasts['selected_minus_self']['matched']['lower_pp']),
            selected_joint_minus_independent_lower_CI_positive=positive(joint['selected']['lower_pp'])),
        independent_confirmation=False,whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,
        new_trajectory_training=False,stage5c_executed=False,smc_enabled=False)
