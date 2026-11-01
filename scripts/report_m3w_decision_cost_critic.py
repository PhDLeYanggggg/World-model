"""Regenerate complete decision-cost evidence from public aggregates only."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_decision_cost_critic as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def build():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest: raise ValueError('Frozen scientific definition changed')
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text()); fits=[]
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']: raise ValueError('Original fit receipt changed')
        fits.append(json.loads((ROOT/ref['path']).read_text()))
    if (len(fits)!=18 or any(v['step']!=2000 or v['parameters']!=30857 or not v['paired_features_initialization_draws_updates_exact']
            or v['unknown_rows_with_supervised_labels'] for v in fits)):
        raise ValueError('All18 complete matched SOURCE fits required')
    cfg=json.loads(run.CONFIG.read_text()); bundle=verify(ROOT,run.PUBLIC)
    groups=aggregate_groups(run.NAME); original=aggregate_groups(run.previous.NAME)
    complete=json.loads((run.PUBLIC/'complete.json').read_text()); replay=json.loads((run.PUBLIC/'verification.json').read_text())
    rows=[groups[Path(ref['path']).name] for ref in complete['groups']]
    summary=json.loads((run.PUBLIC/'summary.json').read_text())
    if (complete['training_freeze_sha256']!=run.sha(run.PUBLIC/'training_freeze.json')
            or complete['summary_sha256']!=run.sha(run.PUBLIC/'summary.json')
            or replay['summary_sha256']!=run.sha(run.PUBLIC/'summary.json')
            or replay['complete_sha256']!=run.sha(run.PUBLIC/'complete.json')
            or not replay['all72_frozen_actions_inference_and_scores_exact'] or not replay['all72_old_controls_exact']):
        raise ValueError('Separate complete verification and frozen training required')
    records=set()
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json'; old=original[key]
        if (row['ids_sha256']!=old['ids_sha256']
                or row['prediction_sha256']['protected_benefit']!=old['prediction_sha256']['released_benefit']
                or row['action_sha256']['protected_joint']!=old['action_sha256']['released_joint']
                or row['result']['policies']['regression_joint']!=old['result']['policies']['released_joint']):
            raise ValueError('Old complete full-policy controls changed')
        records.update((row['source'],rec) for rec in row['result']['recordings'])
    rebuilt=run.evaluation.summarize(rows,reg['contexts'],cfg); rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if rebuilt!=summary or len(records)!=74: raise ValueError('Canonical full cohort summary differs')
    tails={}
    for arm in run.evaluation.NAMES:
        values=[g[arm] for r in rows for g in r['result']['recordings'].values()]
        known=[(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/v['full_known_easy_reference_mass']
            for v in values if v['full_known_easy_reference_mass']>0]
        upper=[v['easy_degradation_upper'] for v in values if v['easy_degradation_upper'] is not None]
        tails[arm]=dict(required_record_views=len(values),known_defined=len(known),
            known_max_degradation_percent=100*max(known) if known else None,known_above2_percent=sum(x>.02+1e-12 for x in known),
            completion_upper_defined=len(upper),completion_upper_max_degradation_percent=100*max(upper) if upper else None,
            completion_upper_above2_percent=sum(x>.02+1e-12 for x in upper))
    fit_stats=dict(complete_fits=18,primary_updates=36000,exact_refit_updates=2000,
        mean_first_monitor={k:sum(v['first_trace']['monitor'][k] for v in fits)/18 for k in ('total','pointwise_risk','selected_query_score','benefit_MSE','decision_cost')},
        mean_final_monitor={k:sum(v['final_trace']['monitor'][k] for v in fits)/18 for k in ('total','pointwise_risk','selected_query_score','benefit_MSE','decision_cost')},
        maximum_dual=max(v['maximum_dual'] for v in fits),
        final_expected_record_constraint_count=sum(v['record_count']*3 for v in fits),
        final_positive_expected_record_constraints=sum(sum(v['final_trace']['full_record_pass']['positive_expected_record_constraints']) for v in fits),
        mean_final_soft_intervention_fraction=sum(v['final_trace']['full_record_pass']['soft_intervention_fraction'] for v in fits)/18,
        full_record_expected_values_are_not_final_hard_policy_risk=True)
    changes={}
    for row in rows:
        result=row['result'];reference=result['contrasts']['joint']['full']['full_reference_mass_counted_once']
        a=result['policies']['regression_joint']['policy'];b=result['policies']['decision_joint']['policy']
        changes.setdefault(row['source'],[]).append([100*(b[k]-a[k])/reference for k in ('selected_known_benefit_mass','selected_known_harm_mass')])
    by_source={k:run.np.mean(x,0).tolist() for k,x in changes.items()}
    if any(abs(v[0]-v[1]-summary['effects']['joint']['known_pp']['by_locality'][k])>1e-8 for k,v in by_source.items()):
        raise ValueError('Original known benefit/harm accounting changed')
    accounting=dict(benefit_delta_pp=float(run.np.mean([x[0] for x in by_source.values()])),
        harm_delta_pp=float(run.np.mean([x[1] for x in by_source.values()])),
        paired_unknown_disagreement_penalty_pp=summary['effects']['joint']['known_pp']['point']-summary['effects']['joint']['lower_pp']['point'],
        by_source_benefit_harm=by_source,descriptive_not_causal_attribution=True,
        calibration_MSE_equal_group_mean={a:{k:sum(r['result']['calibration'][a][k] for r in rows)/72 for k in ('0','1','4')}
            for a in ('regression_benefit','decision_benefit')},
        choice_views={a:{str(k):sum(r['result']['policies'][a]['action_counts'][str(k)] for r in rows) for k in range(-1,7)}
            for a in ('regression_joint','decision_joint')})
    return dict(summary=summary,complete=complete,freeze=freeze,replay=replay,bundle=bundle,
        tails=tails,fit_stats=fit_stats,accounting=accounting,pilot=json.loads((run.PUBLIC/'pilot.json').read_text()),
        full_refit=json.loads((run.PUBLIC/'first_full_refit.json').read_text()))


def cell(value):
    return 'not_estimable' if value is None else f'{value:+.6f}'


def interval(v):
    return 'not_estimable' if v['CI95'] is None else f'{v["point"]:+.6f} [{v["CI95"][0]:+.6f},{v["CI95"][1]:+.6f}]'


def render(v):
    p=run.PUBLIC;s=v['summary'];risk=s['safety']['decision_joint'];primary=s['effects']['joint']['lower_pp']
    run.write(p/'training_statistics.json',v['fit_stats']);run.write(p/'easy_tail_description.json',v['tails'])
    run.write(p/'descriptive_accounting.json',v['accounting'])
    lines=['# Seven-Action Decision-Cost Training: Complete SOURCE Result','','## Material Passport','',
        'fresh_run:18 real Torch cost critics,36,000 primary updates,2,000 exact first-head refit,complete72-group allocation/readout.',
        'cached_verified:18 regression heads,seven forecasts,floor,462 past-only fields and all old full-policy controls.',
        'not_run:new trajectory/JEPA/Transformer training,independent calibration/confirmation and deployment.',
        'All views are exposed SOURCE fitting resubstitution,not whole-pipeline OOF or independent evidence.','',
        '## Registered Factor','',
        'Add the registered soft expected decision-cost loss and full-record risk duals to the original regression loss.',
        'Same30,857 parameters,initialization,queries,draw hashes and2,000 updates. Original reference moments and2%hard decoder stay fixed.',
        'Unknown rows use causal worst-case envelopes in the decision objective,not fabricated supervised labels.',
        'Training optimizes a soft surrogate;actual evaluation uses the unchanged deterministic eligibility/MILP.',
        'The primary is decision-joint minus regression-joint FULL completion-lower ADE utility.','',
        '|Contrast|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|Completion-upper pp [conditional95%CI]|',
        '|---|---:|---:|---:|']
    for mode in ('joint','matched'):
        e=s['effects'][mode];lines.append(f'|{mode}|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    lines+=['','|Policy|Known ADE gain over floor (%)|Own completion-lower pp|Original record risk supported|Known easy-risk violations|Undefined selected easy|Selected/unknown views|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.evaluation.NAMES:
        r=s['safety'][arm]
        lines.append(f'|{arm}|{interval(s["accuracy"][arm])}|{interval(s["floor_lower"][arm])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{r["selected"]}/{r["selected_unknown"]}|')
    a=v['accounting'];lines+=['','## Descriptive Accounting','',
        f'Known benefit change:{a["benefit_delta_pp"]:+.6f}pp;known harm change:{a["harm_delta_pp"]:+.6f}pp.',
        'Benefit minus harm reproduces the paired known utility in every source. This is descriptive,not causal attribution.',
        f'Known-minus-lower paired unknown disagreement penalty:{a["paired_unknown_disagreement_penalty_pp"]:.6f}pp.',
        'Fewer selected unknown rows need not mean less disagreement with the old policy on unobserved outcomes.','',
        '|Moment normalized MSE, supplementary equal-group mean|Regression|Decision cost|','|---|---:|---:|']
    for k,label in (('0','Benefit'),('1','Harm'),('4','Easy harm')):
        lines.append(f'|{label}|{a["calibration_MSE_equal_group_mean"]["regression_benefit"][k]:.6f}|{a["calibration_MSE_equal_group_mean"]["decision_benefit"][k]:.6f}|')
    lines+=['','## Easy Record Tails','','|Policy|Known easy defined|Worst known degradation (%)|Known views above2%|Completion-upper defined|Worst upper bound (%)|',
        '|---|---:|---:|---:|---:|---:|']
    for a in ('regression_joint','decision_joint'):
        t=v['tails'][a];lines.append(f'|{a}|{t["known_defined"]}/{t["required_record_views"]}|{cell(t["known_max_degradation_percent"])}|{t["known_above2_percent"]}|{t["completion_upper_defined"]}|{cell(t["completion_upper_max_degradation_percent"])}|')
    lines+=['','Upper values are conservative missing-outcome bounds,not observed error. Empty risk is not_estimable,not safe0.',
        'Pooled hard/easy gains are supplementary and never replace each original recording requirement.','',
        '## Hard And Easy Known-Label Slices','','|Subset|Regression joint gain (%)|Decision joint gain (%)|','|---|---:|---:|']
    for subset in ('hard','easy'):
        x=s['subsets'][subset]['policies'];lines.append(f'|{subset}|{cell(x["regression_joint"]["ADE_gain_percent"])}|{cell(x["decision_joint"]["ADE_gain_percent"])}|')
    stats=v['fit_stats'];lines+=['','## Real Training And Reproduction','',
        f'Training PID{v["freeze"]["pid"]}: {v["freeze"]["seconds"]:.6f}s,peakRSS{v["freeze"]["peak_RSS_bytes"]}B.',
        f'Readout PID{v["complete"]["pid"]}: {v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B.',
        'These phase timers start at runtime_ready;initial admission is not included.',
        f'Exact first2,000-update refit:{v["full_refit"]["scientific_state_exact"]}. All72 frozen-action/inference/score checks exact:{v["replay"]["all72_frozen_actions_inference_and_scores_exact"]}.',
        'The final verifier does not reoptimize the wall-clock-limited MILPs. Do not call it a fresh allocation replay.',
        f'Final soft expected constraints positive:{stats["final_positive_expected_record_constraints"]}/{stats["final_expected_record_constraint_count"]}.',
        f'Mean final soft intervention fraction:{stats["mean_final_soft_intervention_fraction"]:.6f};maximum dual:{stats["maximum_dual"]:.6f}.',
        'Losses include changing training duals. Their total magnitudes are not directly comparable across updates or proof of final risk.',
        'Native arm64 CPU4/interop1/workers0,checkpoint100/heartbeat50. Full fixed budget,no speed-based downgrade.','',
        '## Fitting Gates','']
    lines += [f'- {k}: {"pass" if value else "fail"}' for k,value in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement_gate"]}. No deployment or independent-role promotion from this run.','',
        '![Decision-cost SOURCE readout](figures/decision_cost_readout.svg)','',
        '## Population And Limits','',
        'All149,510 unique targets,74 recordings,72 source/context/critic-seed groups and12 exposed localities retained.',
        '3,000 paired locality bootstrap draws after averaging repeated context/critic views. Conditional,nominal and adaptively developed,not independent confirmation.',
        'Critic seeds17/29/43 share forecast producer43. Repeated/overlapping windows are not independent examples.',
        'Full-roster FDE remains not_estimable where requested endpoints are absent;no earlier endpoint substitution.',
        'Wholly unknown completion bounds do not resolve missing interior labels.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Stage5C and SMC remain disabled. Not yet a CCF-A submission candidate.','']
    (p/'readout.md').write_text('\n'.join(lines))
    (p/'gates.md').write_text('# Decision-Cost Fitting Gates\n\n'+'\n'.join(f'- {k}: {value}' for k,value in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement_gate"]}. Independent confirmation:false. Submission ready:false. Stage5C:false. SMC:false.\n')
    with (p/'locality_effects.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['source','joint_known_pp','joint_completion_lower_pp'])
        for name in sorted(primary['by_locality']): writer.writerow([name,s['effects']['joint']['known_pp']['by_locality'][name],primary['by_locality'][name]])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),complete_sha256=run.sha(p/'complete.json'),
        training_freeze_sha256=run.sha(p/'training_freeze.json'),verification_sha256=run.sha(p/'verification.json'),
        archive_sha256=v['bundle']['archive_sha256'],new_neural_cost_training=True,new_trajectory_training=False,
        fitting_advancement=s['fitting_advancement_gate'],independent_confirmation=False,deployment_changed=False,submission_ready=False))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    names=('regression_joint','decision_joint');colors=('#326a95','#b85042')
    for i,(name,color) in enumerate(zip(names,colors)):
        z=s['accuracy'][name];ci=z['CI95']
        axes[0].bar(i,z['point'],color=color,width=.6)
        if ci is not None:axes[0].errorbar(i,z['point'],yerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='none',ecolor='#222222',capsize=5)
    axes[0].set_xticks(range(2),['Regression','Decision cost']);axes[0].set_ylabel('Known ADE gain over floor (%)')
    axes[0].set_title('Fixed forecaster, full policies',fontsize=11)
    for i,(key,label) in enumerate((('known_pp','Known labels'),('lower_pp','Completion lower'))):
        z=s['effects']['joint'][key];ci=z['CI95']
        axes[1].errorbar(z['point'],i,xerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='o',color=colors[i],capsize=5)
    axes[1].axvline(0,color='#333333',lw=.8);axes[1].set_yticks([0,1],['Known labels','Completion lower'])
    axes[1].set_xlabel('Decision minus regression (percentage points)');axes[1].set_title('Paired conditional 95% intervals',fontsize=11)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('SOURCE development only: 12 exposed localities, no deployment',fontsize=12)
    dest=run.PUBLIC/'figures';dest.mkdir(exist_ok=True);run.PRIVATE.mkdir(parents=True,exist_ok=True)
    svg=dest/'decision_cost_readout.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'decision_cost_readout.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    value=build();render(value);figure(value);print(json.dumps(dict(groups=value['summary']['groups'],fits=value['fit_stats']['complete_fits'],
        primary=value['summary']['effects']['joint']['lower_pp'],advancement=value['summary']['fitting_advancement_gate'])))
