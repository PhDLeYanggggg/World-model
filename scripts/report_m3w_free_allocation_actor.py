"""Rebuild the frozen actor study from public scalar evidence,not private caches."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_free_allocation_actor as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def training_statistics(fits):
    expected={(c,s,a) for c in json.loads(run.REG.read_text())['contexts']
        for s in (17,29,43) for a in run.api.ARMS}
    if len(fits)!=36 or {(f['context'],f['seed'],f['arm']) for f in fits}!=expected:
        raise ValueError('All36 distinct original actor fits required')
    if any(f['step']!=2000 or f['parameters']!=4321 or f['unknown_rows_with_supervised_labels'] for f in fits):
        raise ValueError('Complete real training without fabricated unknown labels required')
    for c,s,_ in expected:
        pair=[f for f in fits if (f['context'],f['seed'])==(c,s)]
        if pair[0]['input_hashes']!=pair[1]['input_hashes'] or pair[0]['draw_hash']!=pair[1]['draw_hash']:
            raise ValueError('Matched preprocessing,features and training draws required')
    result=dict(complete_fits=36,primary_updates=72000,exact_refit_updates=4000,arms={})
    for arm in run.api.ARMS:
        cells=[f for f in fits if f['arm']==arm];final=[f['final_trace']['full_pass'] for f in cells]
        result['arms'][arm]=dict(fits=18,
            first_monitor_mean={k:sum(f['first_trace']['monitor'][k] for f in cells)/18
                for k in ('total','expected_gain_cost','record_risk_cost','causal_query_risk_cost')},
            final_monitor_mean={k:sum(f['final_trace']['monitor'][k] for f in cells)/18
                for k in ('total','expected_gain_cost','record_risk_cost','causal_query_risk_cost')},
            positive_expected_record_constraints=sum(sum(v['positive_expected_record_constraints']) for v in final),
            positive_predicted_query_constraints=sum(sum(v['positive_predicted_query_constraints']) for v in final),
            soft_intervention_fraction_mean=sum(v['soft_intervention_fraction'] for v in final)/18,
            max_record_dual=max(v['maximum_record_dual'] for v in final),
            max_query_dual=max(v['maximum_query_dual'] for v in final),
            unknown_bound_training_draws=sum(f['unknown_bound_rows_drawn'] for f in cells))
    result['expected_training_constraints_are_not_hard_policy_safety']=True
    return result


def tails_and_accounting(rows,summary):
    tails={}
    for arm in run.evaluation.NAMES:
        values=[g[arm] for r in rows for g in r['result']['recordings'].values()]
        known=[(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/v['full_known_easy_reference_mass']
            for v in values if v['full_known_easy_reference_mass']>0]
        upper=[v['easy_degradation_upper'] for v in values if v['easy_degradation_upper'] is not None]
        tails[arm]=dict(required_record_views=len(values),known_defined=len(known),
            known_max_degradation_percent=100*max(known) if known else None,
            known_above2_percent=sum(v>.02+1e-12 for v in known),
            completion_upper_defined=len(upper),completion_upper_max_degradation_percent=100*max(upper) if upper else None,
            completion_upper_above2_percent=sum(v>.02+1e-12 for v in upper))
    changes={}
    for row in rows:
        result=row['result'];reference=result['comparisons']['selected_minus_zero']['joint']['full']['full_reference_mass_counted_once']
        a=result['policies']['zero_joint']['policy'];b=result['policies']['selected_joint']['policy']
        changes.setdefault(row['source'],[]).append([100*(b[k]-a[k])/reference
            for k in ('selected_known_benefit_mass','selected_known_harm_mass')])
    by_source={k:run.np.mean(v,0).tolist() for k,v in changes.items()}
    effect=summary['contrasts']['selected_minus_zero']['joint']
    for k,v in by_source.items():
        if abs(v[0]-v[1]-effect['known_pp']['by_locality'][k])>1e-8:
            raise ValueError('Known benefit-minus-harm accounting differs')
    return tails,dict(by_source_benefit_harm_delta_pp=by_source,
        mean_benefit_delta_pp=float(run.np.mean([v[0] for v in by_source.values()])),
        mean_harm_delta_pp=float(run.np.mean([v[1] for v in by_source.values()])),
        unknown_disagreement_penalty_pp=effect['known_pp']['point']-effect['lower_pp']['point'],
        choice_views={a:{str(k):sum(r['result']['policies'][a]['action_counts'][str(k)] for r in rows)
            for k in range(-1,7)} for a in run.evaluation.NAMES},
        descriptive_not_causal_attribution=True)


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Frozen scientific definition changed')
    freeze=json.loads((p/'training_freeze.json').read_text());fits=[]
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Original fit receipt changed')
        fit=json.loads((ROOT/ref['path']).read_text())
        if fit['registration_sha256']!=run.sha(run.REG):raise ValueError('Fit registration changed')
        fits.append(fit)
    stats=training_statistics(fits);bundle=verify(ROOT,p)
    complete=json.loads((p/'complete.json').read_text());replay=json.loads((p/'verification.json').read_text())
    if (complete['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
            or freeze['registration_sha256']!=run.sha(run.REG)
            or complete['summary_sha256']!=run.sha(p/'summary.json')
            or replay['summary_sha256']!=run.sha(p/'summary.json')
            or replay['complete_sha256']!=run.sha(p/'complete.json')
            or not replay['all72_frozen_actions_inference_and_scores_exact']
            or not replay['all72_zero_and_legacy_controls_exact'] or replay['optimization_replayed']):
        raise ValueError('Frozen training and separate original-score verification required')
    original=aggregate_groups(run.previous.previous.NAME);legacy=aggregate_groups(run.previous.NAME)
    groups=aggregate_groups(run.NAME);rows=[groups[Path(r['path']).name] for r in complete['groups']]
    records=set()
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json';old=original[key];v1=legacy[key]
        if (row['ids_sha256']!=old['ids_sha256']
                or row['cost_prediction_sha256']!=old['prediction_sha256']['released_benefit']
                or row['action_sha256']['zero_joint']!=old['action_sha256']['released_joint']
                or row['result']['policies']['zero_joint']!=old['result']['policies']['released_joint']
                or row['action_sha256']['decision_v1_joint']!=v1['action_sha256']['released_joint']
                or row['result']['policies']['decision_v1_joint']!=v1['result']['policies']['decision_joint']
                or row['registration_sha256']!=run.sha(run.REG)
                or row['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
                or row['held_scored'] or row['independent_roles_opened']):
            raise ValueError('Complete original cohort,frozen costs and legacy controls required')
        records.update((row['source'],r) for r in row['result']['recordings'])
    summary=json.loads((p/'summary.json').read_text())
    rebuilt=run.evaluation.summarize(rows,reg['contexts'],json.loads(run.CONFIG.read_text()))
    rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if summary!=rebuilt or len(records)!=74:raise ValueError('Complete canonical summary differs')
    refits={a:json.loads((p/f'first_full_refit_{a}.json').read_text()) for a in run.api.ARMS}
    if not all(v['scientific_state_exact'] and v['updates']==2000 for v in refits.values()):
        raise ValueError('Both complete full-refit controls required')
    tails,accounting=tails_and_accounting(rows,summary)
    return dict(summary=summary,complete=complete,freeze=freeze,replay=replay,bundle=bundle,
        fit_stats=stats,tails=tails,accounting=accounting,refits=refits,
        pilot=json.loads((p/'pilot.json').read_text()),decoder_pilot=json.loads((p/'decoder_pilot.json').read_text()))


def cell(v):return 'not_estimable' if v is None else f'{v:+.6f}'


def interval(v):
    return 'not_estimable' if v['CI95'] is None else f'{v["point"]:+.6f} [{v["CI95"][0]:+.6f},{v["CI95"][1]:+.6f}]'


def render(v):
    p=run.PUBLIC;s=v['summary'];primary=s['contrasts']['selected_minus_zero']['joint']
    for name,key in (('training_statistics','fit_stats'),('easy_tail_description','tails'),('descriptive_accounting','accounting')):
        run.write(p/(name+'.json'),v[key])
    lines=['# Free Allocation Actors: Complete SOURCE Readout','','## Material Passport','',
        'fresh_run:36 real Torch actors,72,000 primary updates,4,000 exact-refit updates and complete72-group hard allocation/readout.',
        'cached_verified:18 frozen continuous regression critics,seven forecasts,floor,old full-joint actions and legacy decision-cost actions.',
        'not_run:new forecasting/JEPA/Transformer training,independent calibration/confirmation,deployment and whole-pipeline OOF.',
        'All outcomes are exposed SOURCE fitting/resubstitution,not independent efficacy evidence.','',
        '## Registered Primary And Controls','',
        'Selected-context joint minus zero-actor joint FULL completion-lower ADE utility is the primary.',
        'Self-context is a prespecified matched architecture control,not a replacement primary after seeing outcomes.',
        'Actor offsets are routing scores,not calibrated cost moments or trajectory corrections.',
        'The original frozen2%cost constraints and all seven eligible actions remain.','',
        '|Contrast|Policy|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|Completion-upper pp [conditional95%CI]|',
        '|---|---|---:|---:|---:|']
    for name,modes in s['contrasts'].items():
        for mode,e in modes.items():lines.append(f'|{name}|{mode}|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    e=s['selected_minus_legacy'];lines.append(f'|selected_minus_legacy_decision_v1|joint|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    lines+=['','Count-matched controls test utility only. Empty comparisons remain not_estimable,not safe0.','',
        '|Policy|Known ADE gain over floor (%)|Own completion-lower pp|Original record risk supported|Known easy-risk violations|Undefined selected easy|Selected/unknown views|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.evaluation.NAMES:
        r=s['safety'][arm];lines.append(f'|{arm}|{interval(s["accuracy"][arm])}|{interval(s["floor_lower"][arm])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{r["selected"]}/{r["selected_unknown"]}|')
    lines+=['','## Joint Versus Independent','','|Actor|Known pp|Completion-lower pp|Completion-upper pp|','|---|---:|---:|---:|']
    for arm,e in s['joint_effects'].items():lines.append(f'|{arm}|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    lines+=['','## Easy Recording Tails','','|Policy|Known easy defined|Worst known degradation (%)|Known views above2%|Worst completion-upper bound (%)|',
        '|---|---:|---:|---:|---:|']
    for arm,t in v['tails'].items():lines.append(f'|{arm}|{t["known_defined"]}/{t["required_record_views"]}|{cell(t["known_max_degradation_percent"])}|{t["known_above2_percent"]}|{cell(t["completion_upper_max_degradation_percent"])}|')
    lines+=['','Completion-upper values are conservative missing-outcome bounds,not observed errors.',
        'Pooled hard/easy gains do not replace the original recording-level all/easy requirements.','',
        '|Known-label subset|Zero joint gain (%)|Self joint gain (%)|Selected joint gain (%)|','|---|---:|---:|---:|']
    for subset in ('hard','easy'):
        x=s['subsets'][subset]['policies'];lines.append('|'+subset+'|'+'|'.join(cell(x[a]['ADE_gain_percent']) for a in ('zero_joint','self_joint','selected_joint'))+'|')
    a=v['accounting'];lines+=['','## Descriptive Accounting','',
        f'Benefit delta:{a["mean_benefit_delta_pp"]:+.6f}pp;harm delta:{a["mean_harm_delta_pp"]:+.6f}pp.',
        f'Paired unknown disagreement penalty:{a["unknown_disagreement_penalty_pp"]:.6f}pp.',
        'Benefit minus harm reconstructs every locality known contrast. This is descriptive,not causal attribution.','',
        '## Real Training And Reproduction','',
        f'Training PID{v["freeze"]["pid"]}:{v["freeze"]["seconds"]:.6f}s,peakRSS{v["freeze"]["peak_RSS_bytes"]}B.',
        f'Readout PID{v["complete"]["pid"]}:{v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B.',
        'Phase timers start at runtime_ready and exclude initial admission. Both full2,000-update refits are exact.',
        'All72 frozen-action inference/score checks and zero/legacy full controls agree exactly.',
        'That verification does not rerun the wall-clock-limited optimizer. Actual18-query interruption/recovery was separately tested.',
        'Training losses contain changing duals and are not calibrated risk certificates or cross-update comparable objectives.',
        'Native arm64 CPU4/interop1/workers0,checkpoint100/heartbeat50;no speed-based budget reduction.','',
        '## Fitting Gates','']
    lines += [f'- {k}: {"pass" if value else "fail"}' for k,value in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement_gate"]}.',
        'No deployment or independent-role promotion. Component checks are descriptive SOURCE controls,not independent confirmation.','',
        '![Free actor SOURCE results](figures/free_actor_readout.svg)','',
        '## Population And Limits','',
        'All149,510 unique targets,74 recordings,72 source/context/critic-seed views and12 exposed localities remain.',
        '3,000 paired locality bootstrap draws average repeated context/seeds before resampling. Conditional and nominal,not independent confirmation.',
        'Critic/actor seeds17/29/43 share forecast producer43;overlapping rows are not independent observations.',
        'Full-roster FDE is not_estimable where requested endpoints are absent;no earlier endpoint substitution.',
        'Wholly unknown bounds do not resolve missing interior labels.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Stage5C and SMC remain disabled. Not yet a CCF-A submission candidate.','']
    (p/'readout.md').write_text('\n'.join(lines))
    (p/'gates.md').write_text('# Free Actor Fitting Gates\n\n'+'\n'.join(f'- {k}: {value}' for k,value in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement_gate"]}. Independent confirmation:false. Submission ready:false. Stage5C:false. SMC:false.\n')
    with (p/'locality_effects.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['source','joint_known_pp','joint_completion_lower_pp'])
        for name in sorted(primary['lower_pp']['by_locality']):writer.writerow([name,primary['known_pp']['by_locality'][name],primary['lower_pp']['by_locality'][name]])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),complete_sha256=run.sha(p/'complete.json'),
        training_freeze_sha256=run.sha(p/'training_freeze.json'),verification_sha256=run.sha(p/'verification.json'),
        archive_sha256=v['bundle']['archive_sha256'],new_neural_actor_training=True,new_cost_training=False,new_trajectory_training=False,
        fitting_advancement=s['fitting_advancement_gate'],independent_confirmation=False,deployment_changed=False,submission_ready=False))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors=('#326a95','#9a6a16','#b85042')
    for i,(name,color) in enumerate(zip(('zero_joint','self_joint','selected_joint'),colors)):
        z=s['accuracy'][name];ci=z['CI95'];axes[0].bar(i,z['point'],color=color,width=.6)
        if ci is not None:axes[0].errorbar(i,z['point'],yerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='none',ecolor='#222222',capsize=5)
    axes[0].set_xticks(range(3),['Frozen native','Self context','Selected context'])
    axes[0].set_ylabel('Known ADE gain over floor (%)');axes[0].set_title('Frozen costs and seven forecasts',fontsize=11)
    for i,key in enumerate(('known_pp','lower_pp')):
        z=s['contrasts']['selected_minus_zero']['joint'][key];ci=z['CI95']
        if ci is not None:axes[1].errorbar(z['point'],i,xerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='o',color=colors[i],capsize=5)
    axes[1].axvline(0,color='#333333',lw=.8);axes[1].set_yticks([0,1],['Known labels','Completion lower'])
    axes[1].set_xlabel('Selected-context minus frozen native (pp)');axes[1].set_title('Paired conditional 95% intervals',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('SOURCE fitting only: 12 exposed localities,no deployment',fontsize=12)
    dest=run.PUBLIC/'figures';dest.mkdir(exist_ok=True);svg=dest/'free_actor_readout.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'free_actor_readout.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    value=build();render(value);figure(value)
    print(json.dumps(dict(groups=value['summary']['groups'],fits=value['fit_stats']['complete_fits'],
        primary=value['summary']['contrasts']['selected_minus_zero']['joint']['lower_pp'],advancement=value['summary']['fitting_advancement_gate'])))
