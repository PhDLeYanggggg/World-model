"""Post-readout cost/support calibration; never a new input or model selection."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_supported_global_motion as run
np=run.np


def probability_metrics(probability,observed):
    q,y=np.asarray(probability),np.asarray(observed)
    if (q.shape!=y.shape or q.ndim!=1 or not len(q) or y.dtype!=bool
            or not np.isfinite(q).all() or ((q<0)|(q>1)).any()):
        raise ValueError('Finite probabilities and observed-role labels required')
    bins=[];ece=0.
    ix=np.minimum((q*10).astype(int),9)
    for k in range(10):
        at=ix==k
        p=float(q[at].mean()) if at.any() else None
        truth=float(y[at].mean()) if at.any() else None
        if at.any():ece+=at.mean()*abs(p-truth)
        bins.append(dict(bin=k,count=int(at.sum()),prediction=p,observed=truth))
    return dict(rows=len(q),Brier=float(np.mean((q-y)**2)),ECE=float(ece),
        mean_prediction=float(q.mean()),observed_rate=float(y.mean()),bins=bins)


def main():
    a=run.read(run.PUBLIC/'analysis.json');reg=run.read(run.REG);rows=[];parity=[]
    if a['fresh_models']!=96:raise ValueError('Full immutable readout first')
    for node in reg['jobs']:
        own,old=run.parent.own_and_floor(node);c=run.context(node['job']['key'],own['ids'])
        for seed in (17,29,43):
            for arm in run.method.ARMS:
                key=f'{node["job"]["key"]}_{arm}_seed{seed}'
                pred=run.parent.load_cache(run.read(run.PUBLIC/'inference'/f'{key}.json')['cache'])
                for site in node['part']['names']:
                    at=np.flatnonzero(own['sites']==site);rr,action=own['row'][at],own['action'][at]
                    state=run.method.base.read_checkpoint(run.checked(run.read(run.PUBLIC/'fits'/f'{node["job"]["key"]}_{site}_seed{seed}_{arm}.json')['checkpoint']))
                    if arm=='augmented_local':
                        ref=run.read(run.parent.PUBLIC/'fits'/f'{node["job"]["key"]}_{site}_seed{seed}_role_separated.json')['checkpoint']
                        old_state=run.method.base.read_checkpoint(run.checked(ref))
                        weights=all(run.torch.equal(v,state['model'][k]) for k,v in old_state['model'].items())
                        delta=max(float((v-state['model'][k]).abs().max()) for k,v in old_state['model'].items())
                        parity.append(dict(source=site,seed=seed,final_parent_weights_exact=weights,
                            maximum_absolute_parent_weight_difference=delta,
                            preprocessing_exact=run.stable(state['preprocess']['parent'])==run.stable(old_state['preprocess']),
                            sampler_exact=run.torch.equal(state['generator'],old_state['generator']) and np.array_equal(state['visits'],old_state['visits']),
                            masked_global_weights_zero=all(not bool(v.count_nonzero()) for k,v in state['model'].items() if k.startswith('global_'))))
                    known=own['observed'][at]
                    conditional=pred['conditional'][rr,action]/own['scales'][at,None]
                    mse=np.mean((conditional[known]-own['moments'][at][known])**2,axis=0)
                    unique,first=np.unique(rr,return_index=True)
                    q=pred['probability'][rr[first],action[first]]
                    metrics=probability_metrics(q,known[first])
                    pr=state['preprocess'];z=(c[at][first]-pr['context_mean'])/pr['context_std']
                    policies={}
                    for name in ('independent','joint_free'):
                        choose=pred[name][rr];selected=choose==action
                        picked=selected&known
                        true=own['moments'][at][picked]*own['scales'][at][picked,None]
                        p=pred['moments'][rr[picked],action[picked]]
                        mass=float(p[:,3].sum())
                        policies[name]=dict(selected_known_candidates=int(picked.sum()),
                            predicted_harm=float(p[:,1].sum()),realized_known_harm=float(true[:,1].sum()),
                            predicted_easy_reference=float(p[:,3].sum()),realized_known_easy_reference=float(true[:,3].sum()),
                            predicted_easy_harm=float(p[:,4].sum()),realized_known_easy_harm=float(true[:,4].sum()),
                            known_noneasy_predicted_easy_credit_fraction=(float(p[true[:,3]==0,3].sum()/mass) if mass>0 else None),
                            credit_fraction_not_probability_calibration_proof=True)
                    rows.append(dict(source=site,seed=seed,arm=arm,conditional_relative_MSE=mse.tolist(),
                        MSE_units='squared_history_scale_normalized_error_not_training_RMS',known_candidates=int(known.sum()),
                        distinct_agent_window_rows=len(unique),future_observation=metrics,
                        context_clip_fraction=float((np.abs(z)>8).mean()),context_admitted=arm.endswith('_global'),
                        policies=policies))
    summary={}
    for arm in run.method.ARMS:
        rr=[r for r in rows if r['arm']==arm]
        summary[arm]=dict(conditional_relative_MSE=np.mean([r['conditional_relative_MSE'] for r in rr],axis=0).tolist(),
            observation_Brier=float(np.mean([r['future_observation']['Brier'] for r in rr])),
            observation_ECE=float(np.mean([r['future_observation']['ECE'] for r in rr])),
            context_clip_fraction=float(np.mean([r['context_clip_fraction'] for r in rr])),
            averaging='equal_source_and_risk_seed_not_pooled_candidates')
    if len(parity)!=24:raise ValueError('All24augmented-local replication controls required')
    run.once(run.PUBLIC/'calibration_analysis.json',run.stable(dict(rows=rows,summary=summary,local_parent_replication=parity,
        targets=['benefit','harm','reference','easy_reference','easy_harm'],
        role='posthoc_exposed_own_B_diagnostic_not_new_fit_selection_or_calibration',
        source='fresh_run_from_cached_verified_frozen_predictions',future_observation_not_failure_probability=True,
        independent_confirmation=False,original_rules_unchanged=True,deployment_changed=False)))
    lines=['# Held-Source Cost And Observation Diagnostics','',
        'Posthoc,already-exposed own-B only. No new fit,threshold or calibration.',
        'Conditional MSE uses the common causal history scale,not each arm\'s',
        'different fitting RMS. Values average equally across8sources and3risk seeds.','',
        '| Arm | B MSE | H MSE | R MSE | ER MSE | EH MSE | Observability Brier | Observability ECE | Context clipped fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        values=[*s['conditional_relative_MSE'],s['observation_Brier'],s['observation_ECE'],s['context_clip_fraction']]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','Future observability is annotation support,not trajectory-failure risk.',
        'It is evaluated once per agent-window,not six times per amplitude.',
        'A local arm\'s clipping statistic describes its masked context,not an input',
        'it used. Easy-credit mass on realized nonEasy rows is an attribution,not',
        'by itself proof of an Easy classification or calibration failure.',
        'Selected credit/harm totals are in calibration_analysis.json; they do not',
        'replace the original per-record2%and unknown-completion contract.','',
        '## Local Replication Control','',
        str(sum(r['final_parent_weights_exact'] for r in parity))+'/24augmented-local models exactly recover',
        'the original role-separated parent weights. Masked context is checked as zero;',
        'normalization and samplers are recorded per pair. This is a consistency',
        'check,not independent evidence or a newly discovered method.']
    (run.PUBLIC/'calibration_analysis.md').write_text('\n'.join(lines)+'\n')
    print('Complete cost/observability diagnostic saved without changing models or policy.')


if __name__=='__main__':main()
