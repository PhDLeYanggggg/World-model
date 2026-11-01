"""Complete SOURCE comparison of zero-step cost projection and frozen controls."""
import numpy as np
from src.evaluation import m3w_coupled_benefit_risk as old
from src.evaluation import m3w_reference_support_preflight as support
from src.world_model import m3w_refit_cost_allocation as decoder

NAMES=(*decoder.NAMES,'raw_locked_joint','raw_learned_joint','floor')
COSTS=('raw_locked','raw_learned','projected_locked','projected_learned')


def readout(packet,choices,ranks,allocation,raw,projected,easy_cut,origins,rms):
    choices=dict(choices,floor=np.full(len(packet['ids']),-1,np.int64))
    policies={a:old.bank.policy_metrics(packet,c) for a,c in choices.items()}
    records={str(r):{a:old.bank.legacy.compact_score(packet['y'][at],packet['envelope'][at],c[at])
        for a,c in choices.items()} for r in sorted(set(packet['recordings'])) for at in (packet['recordings']==r,)}
    comparisons={name:old.bank.pair(packet,choices[a],ranks[a],packet,choices[b],ranks[b])
        for name,a,b in (('learned_projection','raw_learned_joint','learned_joint'),
            ('locked_projection','raw_locked_joint','locked_joint'),
            ('projected_joint_minus_independent','learned_independent','learned_joint'))}
    known=np.isfinite(packet['y']).all((1,2));legal=known[:,None]&packet['allowed']
    pred=dict(zip(COSTS,(raw['locked_references'],raw['learned_references'],projected['locked_references'],projected['learned_references'])))
    mse={a:[float(np.square((p[:,:,k]-packet['y'][:,:,k])/packet['scales'][:,None]/rms[k])[legal].mean())
        if legal.any() else None for k in range(5)] for a,p in pred.items()}
    selected={}
    for name,cost in (('raw_locked_joint','raw_locked'),('raw_learned_joint','raw_learned'),
            ('locked_joint','projected_locked'),('learned_joint','projected_learned')):
        at=np.flatnonzero(known&(choices[name]>=0));action=choices[name][at]
        error=(pred[cost][at,action]-packet['y'][at,action])/packet['scales'][at,None]/rms
        selected[name]=dict(known_selected_rows=len(at),normalized_bias=error.mean(0).tolist() if len(at) else None,
            predicted_moments=pred[cost][at,action].sum(0).tolist(),observed_moments=packet['y'][at,action].sum(0).tolist(),
            selected_harm_underprediction_rows=int((error[:,1]<0).sum()),
            selected_easy_harm_underprediction_rows=int((error[:,4]<0).sum()),
            SOURCE_fitting_not_independent_calibration=True)
    diagnostic={a:support.diagnose(packet,p,dict(zero_joint=choices['raw_locked_joint'],selected_joint=choices['raw_learned_joint']))
        for a,p in projected.items()}
    reference=float(packet['floor_ade'][known].sum())
    if reference<=0:raise ValueError('Original known reference support required')
    ade={a:old.bank.legacy.chosen_values(packet['ade'],packet['floor_ade'],c) for a,c in choices.items()}
    return dict(policies=policies,recordings=records,comparisons=comparisons,cost_MSE=mse,selected_cost_diagnostic=selected,
        floor_lower={a:100*v['policy']['selected_net_gain_lower_mass']/reference for a,v in policies.items()},
        pointwise_diagnostic={a:dict(supports=v['supports'],records={r:{k:s['budget_feasible'] for k,s in z.items()}
            for r,z in v['recordings'].items()}) for a,v in diagnostic.items()},
        subsets=old.subsets(packet,choices,easy_cut),allocation=allocation,
        pair_proxy=old.pair_proxy(packet,decoder.base.query_groups(packet['recordings'],packet['frames'],packet['ids']),choices,origins),
        record_tails={str(r):{a:old.tail_stats(v,packet['recordings']==r) for a,v in ade.items()} for r in records},
        evaluation_targets_and_rollouts_unchanged=True,projection_uses_outcomes=False,new_training=False,whole_pipeline_OOF=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,ss in contexts.items() for s in ss for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:
        raise ValueError('All72 original SOURCE groups required')
    interval=lambda get:old.bank.legacy.interval(rows,get,cfg)
    effects={name:{mode:{b:interval(lambda r,n=name,m=mode,b=b:r['comparisons'][n][m][b])
        for b in ('known_pp','lower_pp','upper_pp')} for mode in ('full','matched')}
        for name in ('learned_projection','locked_projection','projected_joint_minus_independent')}
    safety={a:old.bank.safety(rows,a) for a in NAMES}
    floor={a:interval(lambda r,a=a:r['floor_lower'][a]) for a in NAMES}
    coherence={}
    for a in COSTS:
        cells=[r['coherence'][a] for r in rows]
        coherence[a]=dict(legal_action_views=sum(c['legal_action_views'] for c in cells),
            violations={k:sum(c['violations'][k]['above1e_12_normalized_action_views'] for c in cells) for k in cells[0]['violations']},
            reference_spread={k:sum(c['reference_spread'][k]['rows_with_spread_above1e_12'] for c in cells) for k in ('reference','easy_reference')})
    coherent=all(not any(c['violations'].values()) and not any(c['reference_spread'].values())
        for a,c in coherence.items() if a.startswith('projected_'))
    risk=safety['learned_joint'];positive=lambda v:v['CI95'] is not None and v['CI95'][0]>0
    gates=dict(complete72original_groups=True,original36weights_and_full_record_controls_exact=all(r['original_controls_exact'] for r in rows),
        forecast_labels_harm_and_availability_unchanged=all(r['projection_contract_verified'] for r in rows),
        projected_moments_coherent=coherent,
        primary_projection_full_lower_CI_positive=positive(effects['learned_projection']['full']['lower_pp']),
        projected_own_floor_lower_CI_positive=positive(floor['learned_joint']),
        all_original_recording_risk=risk['supported_views']==72 and risk['supported_record_views']==risk['required_record_views'])
    return dict(groups=72,primary='projected_learned_joint_minus_frozen_unprojected_learned_joint_full_completion_lower',
        effects=effects,coherence=coherence,safety=safety,floor_lower=floor,
        accuracy={a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a:interval(lambda r,a=a:r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        cost_MSE={a:{str(k):interval(lambda r,a=a,k=k:r['cost_MSE'][a][k]) for k in range(5)} for a in COSTS},
        subsets=old.pooled_subsets(rows),matched_empty_groups={n:sum(r['result']['comparisons'][n]['matched_empty'] for r in rows) for n in effects},
        fitting_gates=gates,fitting_advancement=all(gates.values()),new_training=False,
        source_role='exposed_SOURCE_fitting_resubstitution_only',independent_confirmation=False,
        whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
