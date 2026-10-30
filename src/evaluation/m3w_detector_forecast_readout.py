"""Producer-excluded SOURCE development metrics; no learned-policy certification."""
import numpy as np
from src.evaluation.m3w_native_metrics import native_errors

ARMS=('disabled','current','noise','history')


def paired_change(old,new,reference,old_prediction,new_prediction,mask):
    known=np.isfinite(old)&np.isfinite(reference)&mask
    np.testing.assert_array_equal(np.isfinite(old),np.isfinite(new))
    unknown=mask&~np.isfinite(old)
    ref=float(reference[known].sum());delta=float((old[known]-new[known]).sum())
    envelope=np.linalg.norm(old_prediction[unknown]-new_prediction[unknown],axis=-1).max(1)
    missing=float(envelope.sum())
    return dict(known_delta_mass=delta,unknown_disagreement_mass=missing,known_reference_mass=ref,
        known_pp=100*delta/ref if ref>0 else None,lower_pp=100*(delta-missing)/ref if ref>0 else None,
        upper_pp=100*(delta+missing)/ref if ref>0 else None,
        known_rows=int(known.sum()),unknown_rows=int(unknown.sum()))


def site_readout(ids,sites,recordings,history,target,valid,physical,reference_index,predictions,*,easy_cut,hard_cut):
    if set(predictions)!=set(ARMS) or len(ids)!=len(set(ids)):raise ValueError('All four frozen input arms and unique fitting IDs required')
    n=len(ids);physical=np.asarray(physical)
    if physical.shape!=(n,6,12,2):raise ValueError('Six original causal physical rollouts required')
    errors={arm:native_errors(predictions[arm],target,valid,np.ones(n)) for arm in ARMS}
    baselines=[native_errors(physical[:,k],target,valid,np.ones(n)) for k in range(6)]
    reference=baselines[reference_index];cv=baselines[1][0]
    oracles={column:np.min(np.stack([p[column] for p in baselines],1),1) for column in (0,1)}
    subsets=dict(all=np.ones(n,bool),hard=np.isfinite(cv)&(cv>=hard_cut),
        easy=np.isfinite(cv)&(cv>0)&(cv<=easy_cut),complete_future=valid.all(1))
    output={}
    for site in sorted(set(sites)):
        group=sites==site;metrics={};contrasts={}
        for subset,base_mask in subsets.items():
            take=group&base_mask
            metrics[subset]={};contrasts[subset]={}
            for metric,column in (('ADE',0),('FDE',1)):
                ref=reference[column];known=take&np.isfinite(ref);den=float(ref[known].sum())
                phys_oracle=oracles[column]
                metrics[subset][metric]=dict(indexed=int(take.sum()),known=int(known.sum()),unknown=int((take&~known).sum()),reference_mass=den,
                    physical={str(k):dict(error_mass=float(p[column][known].sum()),
                        gain_percent=100*(1-float(p[column][known].sum())/den) if den>0 else None) for k,p in enumerate(baselines)},
                    arms={arm:dict(error_mass=float(e[column][known].sum()),
                        gain_percent=100*(1-float(e[column][known].sum())/den) if den>0 else None,
                        positive_harm_mass=float(np.maximum(e[column][known]-ref[known],0).sum()),
                        physical_oracle_plus_neural_gain_percent=100*(1-float(np.minimum(phys_oracle[known],e[column][known]).sum())/den) if den>0 else None)
                        for arm,e in errors.items()},
                    future_informed_oracle_not_deployable=True)
            for control in ARMS:
                contrasts[subset][control]=paired_change(errors[control][0],errors['history'][0],reference[0],
                    predictions[control],predictions['history'],take)
        output[str(site)]=dict(metrics=metrics,contrasts=contrasts,
            records={str(r):dict(rows=int((group&(recordings==r)).sum()),
                known_rows=int((group&(recordings==r)&np.isfinite(reference[0])).sum()),
                native_ADE={arm:dict(mean=float(e[0][group&(recordings==r)&np.isfinite(e[0])].mean())
                    if np.any(group&(recordings==r)&np.isfinite(e[0])) else None,
                    p95=float(np.quantile(e[0][group&(recordings==r)&np.isfinite(e[0])],.95))
                    if np.any(group&(recordings==r)&np.isfinite(e[0])) else None) for arm,e in errors.items()})
                for r in sorted(set(recordings[group]))})
    stationary=np.max(np.abs(history-history[:,-1:]),axis=(1,2))==0
    return dict(rows=n,sites=output,complete_future_rows=int(valid.all(1).sum()),
        partial_future_rows=int((valid.any(1)&~valid.all(1)).sum()),missing_future_rows=int((~valid.any(1)).sum()),
        stationary_observed_rows=int(stationary.sum()),
        stationary_known_nonzero_error_rows={a:int((stationary&np.isfinite(e[0])&(e[0]>0)).sum()) for a,e in errors.items()},
        masked_ADE_not_full_grid_truth_for_partial_rows=True,learned_policy_safety_evaluated=False)


def interval(rows,getter,cfg,roster):
    grouped={str(s):[] for s in roster};seeded={str(s):[] for s in cfg['seeds']}
    for row in rows:
        for site,values in row['result']['sites'].items():
            if site not in grouped:raise ValueError('Unexpected readout locality')
            value=getter(values);grouped[site].append(value);seeded[str(row['seed'])].append(value)
    if any(not v for v in grouped.values()):raise ValueError('Every original readout locality required')
    if any(x is None for values in grouped.values() for x in values):
        return dict(point=None,CI95=None,reason='required_view_not_estimable',
            by_locality={s:float(np.mean(v)) if v and all(x is not None for x in v) else None for s,v in grouped.items()})
    means={s:float(np.mean(v)) for s,v in grouped.items()};v=np.array(list(means.values()))
    draws=np.random.default_rng(cfg['bootstrap_seed']).choice(v,size=(cfg['bootstrap_resamples'],len(v)),replace=True).mean(1)
    return dict(point=float(v.mean()),CI95=np.quantile(draws,[.025,.975]).tolist(),by_locality=means,
        by_seed={s:float(np.mean(v)) for s,v in seeded.items()},bootstrap_resamples=cfg['bootstrap_resamples'],
        bootstrap_unit='SOURCE_locality_after_producer_fold_and_seed_average',independent_confirmation=False)


def summarize(rows,jobs,cfg):
    expected={(r['key'],a) for r in jobs for a in ARMS};keys=[(r['trial'],r['representation']) for r in rows]
    if len(keys)!=72 or len(set(keys))!=72 or set(keys)!=expected:raise ValueError('All72original producer readout groups required')
    roster=sorted({s for r in rows for s in r['result']['sites']})
    if len(roster)!=12:raise ValueError('All12SOURCElocalities required')
    identity={r['key']:r['identity'] for r in jobs}
    for row in rows:
        job=identity[row['trial']]
        if (row['kind']!=job['kind'] or row['seed']!=job['seed']
                or set(row['result']['sites'])!=(set(roster)-set(job['fit_sites']))):
            raise ValueError('Only exact producer-excluded original locality roles may be summarized')
    kinds={}
    for kind in ('single','complement'):
        group=[r for r in rows if r['kind']==kind];representations={}
        for arm in ARMS:
            selected=[r for r in group if r['representation']==arm]
            representations[arm]={subset:{metric:interval(selected,
                lambda v,s=subset,m=metric,a=arm:v['metrics'][s][m]['arms'][a]['gain_percent'],cfg,roster)
                for metric in ('ADE','FDE')} for subset in ('all','hard','easy','complete_future')}
        selected=[r for r in group if r['representation']=='history']
        contrasts={subset:{control:{bound:interval(selected,
            lambda v,s=subset,c=control,b=bound:v['contrasts'][s][c][b],cfg,roster)
            for bound in ('known_pp','lower_pp','upper_pp')} for control in ('disabled','current','noise')}
            for subset in ('all','hard','easy','complete_future')}
        baseline_rows=[r for r in group if r['representation']=='disabled']
        physical={str(k):{metric:interval(baseline_rows,
            lambda v,k=k,m=metric:v['metrics']['all'][m]['physical'][str(k)]['gain_percent'],cfg,roster)
            for metric in ('ADE','FDE')} for k in range(6)}
        kinds[kind]=dict(representations=representations,history_contrasts=contrasts,physical_baselines=physical)
    primary=kinds['single']['history_contrasts']['all']
    positive=lambda x:x['CI95'] is not None and x['CI95'][0]>0
    gates=dict(full72groups=True,disabled_fresh_legacy_exact=all(r['disabled_legacy_exact'] for r in rows),
        **{'history_minus_'+c+'_known_ADE_CI_positive':positive(primary[c]['known_pp']) for c in ('disabled','current','noise')},
        **{'history_minus_'+c+'_completion_lower_CI_positive':positive(primary[c]['lower_pp']) for c in ('disabled','current','noise')})
    return dict(result_source='fresh_SOURCE_producer_excluded_development_only',primary='single_history',kinds=kinds,
        proposal_input_gates=gates,proposal_input_evidence_gate=all(gates.values()),
        learned_policy_safety_evaluated=False,controller_held_outcomes_opened=False,independent_roles_opened=False,
        deployment_changed=False,submission_ready=False,metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
