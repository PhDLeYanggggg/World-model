"""Matched selection-only forecast costs and unchanged recording risk checks."""
import numpy as np
from src.evaluation.m3w_native_metrics import native_errors


def errors(prediction,target,valid):
    return native_errors(prediction,target,valid,np.ones(len(prediction)))


def fraction(numerator,denominator):
    return float(numerator/denominator) if denominator>0 else None


def policy(prediction,floor,cv,target,valid,recordings,*,easy_cut,hard_cut):
    p=np.asarray(prediction);b=np.asarray(floor);mask=np.asarray(valid);rec=np.asarray(recordings)
    pe,pf=errors(p,target,mask);be,bf=errors(b,target,mask);ce,_=errors(cv,target,mask)
    known=mask.any(1);easy=known&(ce>0)&(ce<=easy_cut);hard=known&(ce>hard_cut)
    disagreement=np.linalg.norm(p-b,axis=-1).max(1);selected=disagreement>0
    benefit=np.where(known,np.maximum(be-pe,0),0);harm=np.where(known,np.maximum(pe-be,0),0)
    records={}
    for recording in sorted(set(rec)):
        at=rec==recording;use=at&selected&known;unknown=at&selected&~known
        mass=float(be[use].sum());emass=float(be[use&easy].sum());full_easy=float(be[at&easy].sum())
        unknown_mass=float(disagreement[unknown].sum());hh=float(harm[use].sum());eh=float(harm[use&easy].sum())
        upper=fraction(hh+unknown_mass,mass);eupper=fraction(eh+unknown_mass,emass)
        easy_net=fraction(float((harm[at&easy]-benefit[at&easy]).sum()),full_easy)
        records[str(recording)]=dict(rows=int(at.sum()),selected=int((at&selected).sum()),unknown_selected=int(unknown.sum()),
            known_all_risk=fraction(hh,mass),known_easy_risk=fraction(eh,emass),
            completion_all_risk_upper=upper,completion_easy_risk_upper=eupper,known_easy_net_degradation=easy_net,
            supported=bool(upper is not None and eupper is not None and easy_net is not None
                and upper<=.02 and eupper<=.02 and easy_net<=.02))
    subsets={}
    for name,take in (('all',known),('easy',easy),('hard',hard)):
        subsets[name]=dict(known_rows=int(take.sum()),gain_percent=fraction(float(100*(be[take]-pe[take]).sum()),float(be[take].sum())),
            model_ADE_pixel=float(pe[take].mean()) if take.any() else None,
            floor_ADE_pixel=float(be[take].mean()) if take.any() else None)
    endpoint=mask[:,-1]
    return dict(rows=len(p),known_rows=int(known.sum()),wholly_unknown_rows=int((~known).sum()),
        complete12_rows=int(mask.all(1).sum()),partial_future_rows=int((known&~mask.all(1)).sum()),
        endpoint_rows=int(endpoint.sum()),full_roster_FDE_pixel=float(pf.mean()) if endpoint.all() else None,
        available_endpoint_FDE_diagnostic_pixel=float(pf[endpoint].mean()) if endpoint.any() else None,
        known_ADE_gain_floor_percent=subsets['all']['gain_percent'],
        completion_lower_gain_floor_percent=fraction(float(100*((be[known]-pe[known]).sum()-disagreement[~known].sum())),float(be[known].sum())),
        switch_rate=float(selected.mean()),recordings=records,subsets=subsets,
        p95_known_ADE_pixel=float(np.quantile(pe[known],.95)) if known.any() else None,
        known_ADE_pixel=float(pe[known].mean()) if known.any() else None,
        label_mask_used_for_inference=False,partial_mask_ADE_not_full12_ADE=True)


def paired(disabled,history,floor,target,valid):
    a,_=errors(disabled,target,valid);b,_=errors(history,target,valid);r,_=errors(floor,target,valid)
    known=np.asarray(valid).any(1);den=float(r[known].sum());value=float((a[known]-b[known]).sum())
    distance=float(np.linalg.norm(np.asarray(history)-disabled,axis=-1).max(1)[~known].sum())
    return dict(known_pp=fraction(100*value,den),completion_lower_pp=fraction(100*(value-distance),den),
        completion_upper_pp=fraction(100*(value+distance),den),unknown_path_disagreement=distance,
        required_rows=len(valid),known_rows=int(known.sum()))


def interval(rows,getter,*,expected_sites,resamples,seed):
    values={}
    for row in rows:
        value=getter(row)
        if value is None:return dict(point=None,CI95=None,status='required_view_not_estimable')
        values.setdefault(row['site'],[]).append(value)
    if set(values)!=set(expected_sites):raise ValueError('All six fixed selection localities required')
    means={k:float(np.mean(values[k])) for k in sorted(values)};x=np.asarray(list(means.values()))
    draws=np.random.default_rng(seed).choice(x,size=(resamples,len(x))).mean(1)
    return dict(point=float(x.mean()),CI95=np.quantile(draws,[.025,.975]).tolist(),by_locality=means,
        bootstrap_unit='selection_locality_after_source_fold_and_training_seed_average',bootstrap_resamples=resamples,
        status='model_selection_development_not_final_confirmation')


def summarize(rows,sites,*,resamples,seed):
    expected={(f'single{f}_seed{s}',site) for f in range(3) for s in (17,29,43) for site in sites}
    if len(rows)!=54 or {(r['trial'],r['site']) for r in rows}!=expected:raise ValueError('All nine producers by six selection localities required')
    ci=lambda fn:interval(rows,fn,expected_sites=sites,resamples=resamples,seed=seed)
    policies=tuple(rows[0]['policies']);accuracy={p:ci(lambda r,p=p:r['policies'][p]['known_ADE_gain_floor_percent']) for p in policies}
    lower={p:ci(lambda r,p=p:r['policies'][p]['completion_lower_gain_floor_percent']) for p in policies}
    safety={}
    for p in policies:
        records=[v for r in rows for v in r['policies'][p]['recordings'].values()]
        easy=[v['known_easy_net_degradation'] for v in records if v['known_easy_net_degradation'] is not None]
        safety[p]=dict(required_record_views=len(records),supported_record_views=sum(v['supported'] for v in records),
            known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk']>.02 for v in records),
            known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in records),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in records),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    primary={k:ci(lambda r,k=k:r['paired_history_minus_disabled'][k]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    positive=lambda r:r['CI95'] is not None and r['CI95'][0]>0
    return dict(groups=54,unique_target_rows=38102,unique_recordings=28,selection_localities=6,
        actual_forecaster_training_seeds=[17,29,43],source_folds=3,
        accuracy=accuracy,completion_lower=lower,safety=safety,primary=primary,
        primary_lower_positive=positive(primary['completion_lower_pp']),
        history_floor_lower_positive=positive(lower['history']),
        history_original_record_risk_pass=safety['history']['supported_record_views']==safety['history']['required_record_views'],
        primary_contrast='history_minus_disabled_full_paired_completion_lower',
        model_selection_outcomes_opened=True,training_or_thresholds_changed=False,
        calibration_access=False,confirmation_access=False,independent_final_confirmation=False,
        new_training=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
