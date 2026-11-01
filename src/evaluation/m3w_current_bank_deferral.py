"""Native partial-label scores for current-bank ADE and TDEF adapters."""
import numpy as np
from src.evaluation.m3w_own_decision_cost import old
from src.world_model.m3w_frame_action_allocation import query_groups

NAMES = ('ade_raw','ade_guarded','tdef_raw','tdef_guarded','frozen_own_independent','frozen_own_joint','floor')
PAIRS = (('guarded_tdef_minus_ade','ade_guarded','tdef_guarded'),
         ('guarded_tdef_minus_frozen','frozen_own_joint','tdef_guarded'),
         ('raw_tdef_minus_ade','ade_raw','tdef_raw'),
         ('tdef_guarded_minus_raw','tdef_raw','tdef_guarded'))


def readout(packet, choices, ranks, easy_cut, origins):
    choices=dict(choices,floor=np.full(len(packet['ids']),-1,np.int64))
    policies={a:old.bank.policy_metrics(packet,c) for a,c in choices.items()}
    reference=float(packet['floor_ade'][np.isfinite(packet['floor_ade'])].sum())
    if reference<=0 or set(choices)!=set(NAMES):
        raise ValueError('Full registered policies and positive known reference required')
    records={str(r):{a:old.bank.legacy.compact_score(packet['y'][at],packet['envelope'][at],c[at]) for a,c in choices.items()}
        for r in sorted(set(packet['recordings'])) for at in (packet['recordings']==r,)}
    comparisons={n:old.bank.pair(packet,choices[a],ranks[a],packet,choices[b],ranks[b]) for n,a,b in PAIRS}
    ade={a:old.bank.legacy.chosen_values(packet['ade'],packet['floor_ade'],c) for a,c in choices.items()}
    return dict(policies=policies,recordings=records,comparisons=comparisons,
        floor_lower={a:100*p['policy']['selected_net_gain_lower_mass']/reference for a,p in policies.items()},
        subsets=old.subsets(packet,choices,easy_cut),
        pair_proxy=old.pair_proxy(packet,query_groups(packet['recordings'],packet['frames'],packet['ids']),choices,origins),
        record_tails={str(r):{a:old.tail_stats(v,packet['recordings']==r) for a,v in ade.items()} for r in records},
        evaluation_targets_and_rollouts_unchanged=True,inference_uses_outcomes=False,new_trajectory_training=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,ss in contexts.items() for s in ss for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:
        raise ValueError('All72original source groups required')
    interval=lambda fn:old.bank.legacy.interval(rows,fn,cfg)
    effects={n:{m:{k:interval(lambda r,n=n,m=m,k=k:r['comparisons'][n][m][k]) for k in ('known_pp','lower_pp','upper_pp')}
                for m in ('full','matched')} for n,_,_ in PAIRS}
    safety={a:old.bank.safety(rows,a) for a in NAMES}
    floor={a:interval(lambda r,a=a:r['floor_lower'][a]) for a in NAMES}
    positive=lambda x:x['CI95'] is not None and x['CI95'][0]>0
    risk=safety['tdef_guarded']
    gates=dict(complete36fits72groups=True,matched_capacity_initial_draws_budget=all(r['matched_training'] for r in rows),
        complete_query_coverage=all(r['minimum_gradient_query_visits']>=4 for r in rows),
        original_controls_exact=all(r['original_control_exact'] for r in rows),
        causal_frozen_risk_guard_pass=all(r['guarded_predicted_feasible'] for r in rows),
        primary_tdef_minus_ade_full_lower_CI_positive=positive(effects['guarded_tdef_minus_ade']['full']['lower_pp']),
        tdef_minus_frozen_full_lower_CI_positive=positive(effects['guarded_tdef_minus_frozen']['full']['lower_pp']),
        tdef_floor_lower_CI_positive=positive(floor['tdef_guarded']),
        all_original_recording_risk=risk['supported_views']==72 and risk['supported_record_views']==risk['required_record_views'])
    return dict(groups=72,new_fits=36,primary='guarded_tdef_minus_ade_full_completion_lower',effects=effects,
        accuracy={a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a:interval(lambda r,a=a:r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        floor_lower=floor,safety=safety,subsets=old.pooled_subsets(rows),
        matched_empty_groups={n:sum(r['result']['comparisons'][n]['matched_empty'] for r in rows) for n,_,_ in PAIRS},
        fitting_gates=gates,fitting_advancement=all(gates.values()),source_role='exposed_SOURCE_fitting_resubstitution_only',
        independent_confirmation=False,whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,
        new_trajectory_training=False,stage5c_executed=False,smc_enabled=False)
