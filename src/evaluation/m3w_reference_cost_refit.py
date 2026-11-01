"""Complete SOURCE cost-refit readout; oracle support never becomes model utility."""
import numpy as np
from src.world_model import m3w_refit_cost_allocation as decoder
from src.evaluation import m3w_coupled_benefit_risk as old
from src.evaluation import m3w_reference_support_preflight as support

NAMES=(*decoder.NAMES,'parent_joint','floor')


def readout(packet,choices,ranks,info,predictions,easy_cut,origins,rms):
    choices=dict(choices,floor=np.full(len(packet['ids']),-1,np.int64));policies={a:old.bank.policy_metrics(packet,c) for a,c in choices.items()}
    records={str(rec):{a:old.bank.legacy.compact_score(packet['y'][at],packet['envelope'][at],c[at]) for a,c in choices.items()}
        for rec in sorted(set(packet['recordings'])) for at in (packet['recordings']==rec,)}
    contrasts={m:old.bank.pair(packet,choices['locked_'+m],ranks['locked_'+m],packet,choices['learned_'+m],ranks['learned_'+m]) for m in ('joint','matched')}
    known=np.isfinite(packet['y']).all((1,2));legal=known[:,None]&packet['allowed'];reference=float(packet['floor_ade'][known].sum())
    mse={a:[float(np.square((p[:,:,k]-packet['y'][:,:,k])/packet['scales'][:,None]/rms[k])[legal].mean()) if legal.any() else None for k in range(5)] for a,p in predictions.items()}
    diagnostic={a:support.diagnose(packet,p,{key:choices['parent_joint'] for key in support.CONTROLS}) for a,p in predictions.items()}
    ade={a:old.bank.legacy.chosen_values(packet['ade'],packet['floor_ade'],c) for a,c in choices.items()}
    return dict(policies=policies,recordings=records,contrasts=contrasts,calibration_MSE=mse,
        original_parent_effect=old.bank.cross_bank_bounds(packet,choices['parent_joint'],packet,choices['learned_joint']),
        pointwise_diagnostic={a:dict(supports=v['supports'],records={r:{k:s['budget_feasible'] for k,s in values.items()} for r,values in v['recordings'].items()}) for a,v in diagnostic.items()},
        floor_lower={a:100*v['policy']['selected_net_gain_lower_mass']/reference for a,v in policies.items()},
        subsets=old.subsets(packet,choices,easy_cut),allocation=info,
        pair_proxy=old.pair_proxy(packet,decoder.base.query_groups(packet['recordings'],packet['frames'],packet['ids']),choices,origins),
        record_tails={str(r):{a:old.tail_stats(e,packet['recordings']==r) for a,e in ade.items()} for r in records},
        original_evaluation_targets_and_reference_unchanged=True,inference_outcomes_used=False,whole_pipeline_OOF=False,new_trajectory_training=False)


def summarize(rows,contexts,cfg):
    expected={(c,s,k) for c,ss in contexts.items() for s in ss for k in cfg['seeds']}
    if len(rows)!=72 or {(r['context'],r['source'],r['seed']) for r in rows}!=expected:raise ValueError('Full72SOURCE groups required')
    interval=lambda fn:old.bank.legacy.interval(rows,fn,cfg)
    effects={m:{b:interval(lambda r,m=m,b=b:r['contrasts'][m]['full'][b]) for b in ('known_pp','lower_pp','upper_pp')} for m in ('joint','matched')}
    calibration={a:{str(k):interval(lambda r,a=a,k=k:r['calibration_MSE'][a][k]) for k in range(5)} for a in decoder.ARMS}
    reference_change=interval(lambda r:None if any(r['calibration_MSE'][a][k] is None for a in decoder.ARMS for k in (2,3)) else
        float(np.mean(np.array(r['calibration_MSE']['learned_references'])[[2,3]]-np.array(r['calibration_MSE']['locked_references'])[[2,3]])))
    safety={a:old.bank.safety(rows,a) for a in NAMES};floor={a:interval(lambda r,a=a:r['floor_lower'][a]) for a in NAMES};risk=safety['learned_joint']
    positive=lambda v:v['CI95'] is not None and v['CI95'][0]>0
    gates=dict(complete36fits72groups=True,matched_warm_start_draws_and_budget=all(r['matched_training'] for r in rows),
        all_queries_covered_at_least4times=all(r['minimum_gradient_query_visits']>=4 for r in rows),
        original_parent_control_exact=all(r['original_parent_exact'] for r in rows),
        primary_learned_minus_locked_lower_CI_positive=positive(effects['joint']['lower_pp']),
        reference_MSE_decreased_CI_supported=reference_change['CI95'] is not None and reference_change['CI95'][1]<0,
        own_floor_lower_CI_positive=positive(floor['learned_joint']),
        complete_original_recording_risk=risk['supported_views']==72 and risk['supported_record_views']==risk['required_record_views'])
    return dict(groups=72,new_cost_fits=36,effects=effects,calibration_MSE=calibration,reference_MSE_change=reference_change,
        parent_effect={b:interval(lambda r,b=b:r['original_parent_effect'][b]) for b in ('known_pp','lower_pp','upper_pp')},
        accuracy={a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a:interval(lambda r,a=a:r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        safety=safety,floor_lower=floor,subsets=old.pooled_subsets(rows),fitting_gates=gates,fitting_advancement=all(gates.values()),
        independent_confirmation=False,whole_pipeline_OOF=False,deployment_changed=False,submission_ready=False,
        new_trajectory_training=False,stage5c_executed=False,smc_enabled=False)
