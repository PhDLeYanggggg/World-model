"""Complete SOURCE evaluation of matched fixed-choice and own-choice critics."""
import numpy as np
from src.world_model import m3w_own_decision_cost as api
from src.evaluation import m3w_coupled_benefit_risk as old
from src.evaluation import m3w_reference_support_preflight as support

NAMES=(*api.NAMES,'frozen_projected_joint','floor')


def readout(packet,choices,ranks,info,predictions,easy_cut,origins,rms):
    choices=dict(choices,floor=np.full(len(packet['ids']),-1,np.int64))
    policies={a:old.bank.policy_metrics(packet,c) for a,c in choices.items()}
    records={str(r):{a:old.bank.legacy.compact_score(packet['y'][at],packet['envelope'][at],c[at]) for a,c in choices.items()}
        for r in sorted(set(packet['recordings'])) for at in (packet['recordings']==r,)}
    pairs=(('own_minus_fixed','fixed_joint','own_joint'),('own_minus_frozen','frozen_projected_joint','own_joint'),
        ('own_joint_minus_independent','own_independent','own_joint'))
    comparisons={name:old.bank.pair(packet,choices[a],ranks[a],packet,choices[b],ranks[b]) for name,a,b in pairs}
    known=np.isfinite(packet['y']).all((1,2));legal=known[:,None]&packet['allowed'];reference=float(packet['floor_ade'][known].sum())
    if reference<=0:raise ValueError('Original observed reference support required')
    mse={a:[float(np.square((p[:,:,k]-packet['y'][:,:,k])/packet['scales'][:,None]/rms[k])[legal].mean()) if legal.any() else None for k in range(5)] for a,p in predictions.items()}
    selected={}
    for name,arm in (('fixed_joint','fixed_choice'),('own_joint','own_choice')):
        at=np.flatnonzero(known&(choices[name]>=0));action=choices[name][at]
        error=(predictions[arm][at,action]-packet['y'][at,action])/packet['scales'][at,None]/rms
        selected[name]=dict(known_selected_rows=len(at),normalized_bias=error.mean(0).tolist() if len(at) else None,
            harm_underprediction_rows=int((error[:,1]<0).sum()),easy_harm_underprediction_rows=int((error[:,4]<0).sum()))
    diagnostic={a:support.diagnose(packet,p,{key:choices['frozen_projected_joint'] for key in support.CONTROLS}) for a,p in predictions.items()}
    ade={a:old.bank.legacy.chosen_values(packet['ade'],packet['floor_ade'],c) for a,c in choices.items()}
    return dict(policies=policies,recordings=records,comparisons=comparisons,cost_MSE=mse,selected_cost_diagnostic=selected,
        floor_lower={a:100*v['policy']['selected_net_gain_lower_mass']/reference for a,v in policies.items()},
        pointwise_diagnostic={a:dict(supports=v['supports']) for a,v in diagnostic.items()},
        subsets=old.subsets(packet,choices,easy_cut),allocation=info,
        pair_proxy=old.pair_proxy(packet,api.allocation.base.query_groups(packet['recordings'],packet['frames'],packet['ids']),choices,origins),
        record_tails={str(r):{a:old.tail_stats(v,packet['recordings']==r) for a,v in ade.items()} for r in records},
        evaluation_targets_and_rollouts_unchanged=True,inference_uses_outcomes=False,new_trajectory_training=False,whole_pipeline_OOF=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,ss in contexts.items() for s in ss for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:raise ValueError('All72 original SOURCE groups required')
    interval=lambda fn:old.bank.legacy.interval(rows,fn,cfg)
    names=('own_minus_fixed','own_minus_frozen','own_joint_minus_independent')
    effects={n:{m:{k:interval(lambda r,n=n,m=m,k=k:r['comparisons'][n][m][k]) for k in ('known_pp','lower_pp','upper_pp')} for m in ('full','matched')} for n in names}
    safety={a:old.bank.safety(rows,a) for a in NAMES};floor={a:interval(lambda r,a=a:r['floor_lower'][a]) for a in NAMES}
    risk=safety['own_joint'];positive=lambda x:x['CI95'] is not None and x['CI95'][0]>0
    coherent=all(not any(c['violations'][k]['above1e_12_normalized_action_views'] for k in c['violations'])
        and not any(c['reference_spread'][k]['rows_with_spread_above1e_12'] for k in ('reference','easy_reference')) for r in rows for c in r['coherence'].values())
    gates=dict(complete36fits72groups=True,matched_active_capacity_warm_draws_and_budget=all(r['matched_training'] for r in rows),
        all_queries_covered_at_least4times=all(r['minimum_gradient_query_visits']>=4 for r in rows),
        frozen_projected_control_exact=all(r['original_control_exact'] for r in rows),coherent_cost_moments=coherent,
        primary_own_minus_fixed_full_lower_CI_positive=positive(effects['own_minus_fixed']['full']['lower_pp']),
        own_minus_frozen_full_lower_CI_positive=positive(effects['own_minus_frozen']['full']['lower_pp']),
        own_floor_lower_CI_positive=positive(floor['own_joint']),
        all_original_recording_risk=risk['supported_views']==72 and risk['supported_record_views']==risk['required_record_views'])
    return dict(groups=72,new_cost_fits=36,primary='own_choice_minus_fixed_choice_joint_full_completion_lower',effects=effects,
        accuracy={a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a:interval(lambda r,a=a:r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        cost_MSE={a:{str(k):interval(lambda r,a=a,k=k:r['cost_MSE'][a][k]) for k in range(5)} for a in api.ARMS},
        safety=safety,floor_lower=floor,subsets=old.pooled_subsets(rows),
        matched_empty_groups={n:sum(r['result']['comparisons'][n]['matched_empty'] for r in rows) for n in names},
        fitting_gates=gates,fitting_advancement=all(gates.values()),source_role='exposed_SOURCE_fitting_resubstitution_only',
        independent_confirmation=False,whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,
        new_trajectory_training=False,stage5c_executed=False,smc_enabled=False)
