"""All36active-producer paired readouts; no learned deployment policy claim."""
import numpy as np
from src.evaluation import m3w_detector_forecast_readout as old
from src.evaluation.m3w_native_metrics import native_errors

ARMS=old.ARMS


def site_readout(ids,sites,recordings,history,target,valid,physical,reference_index,predictions,*,easy_cut,hard_cut):
    result=old.site_readout(ids,sites,recordings,history,target,valid,physical,reference_index,predictions,easy_cut=easy_cut,hard_cut=hard_cut)
    costs=[native_errors(physical[:,k],target,valid,np.ones(len(ids)))[0] for k in range(6)]
    current=native_errors(predictions['history'],target,valid,np.ones(len(ids)))[0];cv=costs[1]
    subsets=dict(all=np.ones(len(ids),bool),hard=np.isfinite(cv)&(cv>=hard_cut),
        easy=np.isfinite(cv)&(cv>0)&(cv<=easy_cut),complete_future=valid.all(1))
    for site,view in result['sites'].items():
        group=sites==site
        view['history_vs_physical']={sub:{str(k):old.paired_change(costs[k],current,costs[reference_index],
            physical[:,k],predictions['history'],group&mask) for k in range(6)} for sub,mask in subsets.items()}
    return result


def summarize(rows,jobs,cfg):
    expected={(j['key'],a) for j in jobs for a in ARMS};keys=[(r['trial'],r['representation']) for r in rows]
    if len(jobs)!=9 or len(keys)!=36 or len(set(keys))!=36 or set(keys)!=expected:raise ValueError('All9actual parents and36groups required')
    roster=sorted({s for r in rows for s in r['result']['sites']})
    if len(roster)!=12:raise ValueError('Complete12locality roster required')
    identities={j['key']:j['identity'] for j in jobs}
    for row in rows:
        j=identities[row['trial']]
        if (j['kind']!='single' or row['kind']!='single' or row['seed']!=j['seed']
                or set(row['result']['sites'])!=set(roster)-set(j['fit_sites'])):
            raise ValueError('Exact active producer exclusion and seed roles required')
    subsets=('all','hard','easy','complete_future');metrics=('ADE','FDE');bounds=('known_pp','lower_pp','upper_pp')
    interval=lambda selected,getter:old.interval(selected,getter,cfg,roster)
    representations={}
    for arm in ARMS:
        selected=[r for r in rows if r['representation']==arm]
        representations[arm]={sub:{metric:interval(selected,lambda v,s=sub,m=metric,a=arm:v['metrics'][s][m]['arms'][a]['gain_percent'])
            for metric in metrics} for sub in subsets}
    selected=[r for r in rows if r['representation']=='history']
    contrasts={sub:{control:{b:interval(selected,lambda v,s=sub,c=control,b=b:v['contrasts'][s][c][b]) for b in bounds}
        for control in ('disabled','current','noise')} for sub in subsets}
    physical_contrasts={sub:{str(k):{b:interval(selected,lambda v,s=sub,k=str(k),b=b:v['history_vs_physical'][s][k][b]) for b in bounds}
        for k in range(6)} for sub in subsets}
    baseline_rows=[r for r in rows if r['representation']=='disabled']
    physical={str(k):{metric:interval(baseline_rows,lambda v,k=str(k),m=metric:v['metrics']['all'][m]['physical'][k]['gain_percent'])
        for metric in metrics} for k in range(6)}
    positive=lambda v:v['CI95'] is not None and v['CI95'][0]>0
    gates=dict(all36groups=True,all9fresh_parent_predictions_exact=all(r['disabled_legacy_exact'] for r in rows),
        **{'history_minus_'+c+'_known_CI_positive':positive(contrasts['all'][c]['known_pp']) for c in ('disabled','current','noise')},
        **{'history_minus_'+c+'_completion_lower_CI_positive':positive(contrasts['all'][c]['lower_pp']) for c in ('disabled','current','noise')})
    return dict(result_source='fresh_run_active_producer_excluded_SOURCE_development',primary='single_history',
        representations=representations,history_contrasts=contrasts,physical_baselines=physical,
        history_vs_fixed_physical=physical_contrasts,physical_comparisons_preregistered=True,
        physical_known_CI_positive={k:positive(v['known_pp']) for k,v in physical_contrasts['all'].items()},
        physical_completion_lower_CI_positive={k:positive(v['lower_pp']) for k,v in physical_contrasts['all'].items()},
        proposal_input_gates=gates,proposal_input_evidence_gate=all(gates.values()),
        learned_policy_safety_evaluated=False,controller_held_outcomes_opened=False,independent_roles_opened=False,
        deployment_changed=False,submission_ready=False,metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
