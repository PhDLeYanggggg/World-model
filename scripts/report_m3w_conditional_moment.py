"""Reconstruct coherent conditional-cost results from public scalar evidence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_conditional_moment as run
from scripts import report_m3w_boundary_guard as previous
from src.evaluation.m3w_metric_bundle import verify

cell,interval=previous.cell,previous.interval


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text());cfg=json.loads(run.CONFIG.read_text())
    for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Registered definition changed')
    for name,digest in reg['parent_files'].items():
        if run.sha(run.prior.PUBLIC/name)!=digest:raise ValueError('Parent evidence changed')
    previous.build();bundle=verify(ROOT,p)
    freeze=json.loads((p/'training_freeze.json').read_text());complete=json.loads((p/'complete.json').read_text());replay=json.loads((p/'verification.json').read_text())
    if (complete['training_freeze_sha256']!=run.sha(p/'training_freeze.json') or complete['summary_sha256']!=run.sha(p/'summary.json')
            or replay['summary_sha256']!=complete['summary_sha256'] or replay['complete_sha256']!=run.sha(p/'complete.json')
            or not replay['all72_new_moment_probability_inference_and_action_scores_exact']):raise ValueError('Complete real inference replay required')
    fits={}
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text());key=r['context'],r['seed'],r['arm']
        if(key in fits or r['step']!=r['budget'] or r['parameters']!=31381 or r['minimum_query_visits']<4
            or r['unknown_rows_with_supervised_labels'] or r['registration_sha256']!=run.sha(run.REG)):raise ValueError('Complete matched real training required')
        fits[key]=r
    if set(fits)!={(c,s,a) for c in reg['contexts'] for s in reg['seeds'] for a in run.api.ARMS}:raise ValueError('All36heads required')
    for c in reg['contexts']:
        for s in reg['seeds']:
            a,b=(fits[c,s,k] for k in run.api.ARMS)
            for k in ('input_hashes','draw_hash','budget','source_query_counts','warm_checkpoint','row_draws','unknown_rows_drawn','priors'):
                if a[k]!=b[k]:raise ValueError('Paired design changed')
    refits=[json.loads((p/f'first_full_refit_{a}.json').read_text()) for a in run.api.ARMS]
    if(sum(r['step'] for r in fits.values())!=208320 or freeze['primary_optimizer_updates']!=208320
        or sum(r['updates'] for r in refits)!=14336 or freeze['exact_full_refit_updates']!=14336
        or not all(r['scientific_state_exact'] for r in refits)):raise ValueError('Full exact budgets required')
    groups=run.prior.prior.parent.aggregate_groups(run.NAME);old=run.prior.prior.parent.aggregate_groups(run.prior.NAME)
    rows=[groups[Path(ref['path']).name] for ref in complete['groups']];records=set()
    for r in rows:
        prior=old[f'{r["context"]}_{r["source"]}_seed{r["seed"]}_fit.json']
        if(r['registration_sha256']!=run.sha(run.REG) or r['ids_sha256']!=prior['ids_sha256']
            or r['training_freeze_sha256']!=run.sha(p/'training_freeze.json') or r['parent_complete_sha256']!=run.sha(run.prior.PUBLIC/'complete.json')):raise ValueError('Original cohort/lineage changed')
        for a,b in run.CONTROLS.items():
            if r['action_sha256'][a]!=prior['action_sha256'][b] or r['result']['policies'][a]!=prior['result']['policies'][b]:raise ValueError('Original full control changed')
            for rec,v in r['result']['recordings'].items():
                if v[a]!=prior['result']['recordings'][rec][b]:raise ValueError('Original recording control changed')
        records.update((r['source'],k) for k in r['result']['recordings'])
    summary=run.evaluation.summarize(rows,reg['contexts'],cfg);summary.update(unique_target_rows=149510,unique_recordings=len(records))
    if len(records)!=74 or summary!=json.loads((p/'summary.json').read_text()):raise ValueError('Original complete canonical summary required')
    traces=json.loads((p/'training_traces.json').read_text());seen=set()
    if traces['training_freeze_sha256']!=run.sha(p/'training_freeze.json'):raise ValueError('Curve source changed')
    for r in traces['cells']:
        key=r['context'],r['seed'],r['arm']
        if(key in seen or key not in fits or r['checkpoint_sha256']!=fits[key]['checkpoint']['sha256']
            or r['trace'][0]['step']!=0 or r['trace'][-1]['step']!=fits[key]['budget']):raise ValueError('Full real curves required')
        seen.add(key)
    if seen!=set(fits):raise ValueError('All36curves required')
    auxiliary={}
    for a in run.evaluation.NAMES:
        policies=[r['result']['policies'][a] for r in rows];cells=[v[a] for r in rows for v in r['result']['recordings'].values()]
        counts={str(k):sum(v['action_counts'][str(k)] for v in policies) for k in range(-1,7)};total=sum(counts.values())
        easy=[100*(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/v['full_known_easy_reference_mass'] for v in cells if v['full_known_easy_reference_mass']>0]
        auxiliary[a]=dict(action_counts=counts,repeated_row_views=total,switch_rate=1-counts['-1']/total,neural_row_rate=counts['6']/total,
            worst_known_easy_degradation_percent=max(easy) if easy else None,easy_views_above2=sum(v>2+1e-10 for v in easy))
    diagnostic_effects=dict(easy_occurrence_Brier=run.evaluation.old.bank.legacy.interval(rows,lambda r:r['easy_occurrence_Brier']['identified']-r['easy_occurrence_Brier']['marginal'],cfg),
        normalized_moment_MSE={str(k):run.evaluation.old.bank.legacy.interval(rows,lambda r,k=k:r['normalized_moment_MSE']['identified'][k]-r['normalized_moment_MSE']['marginal'][k],cfg) for k in range(5)})
    return dict(summary=summary,training=freeze,fits=list(fits.values()),complete=complete,replay=replay,bundle=bundle,
        traces=traces,auxiliary=auxiliary,diagnostic_effects=diagnostic_effects,pilot=json.loads((p/'pilot.json').read_text()))


def render(v):
    p=run.PUBLIC;s=v['summary'];a=v['auxiliary'];lines=['# Coherent Conditional Cost Identification','',
        '## Material Passport','',
        'fresh_run:36real Torch cost fits,two exact complete refits,full72-group readout and separate full moment/probability inference/action-score replay.',
        'cached_verified:all current forecast paths,TDEF ranking,462causal fields,normalizers,warm encoder and original complete controls.',
        'not_run:new trajectory/JEPA/Transformer training,independent role access,calibration certification or deployment.',
        'This is exposed SOURCE fitting/resubstitution. Three cost seeds share forecaster43.','',
        '## Controlled Change','',
        'Both31,381-parameter models share the same conditional-cost factorization,encoder,initial fitting priors,draws and complete budget.',
        'The identified arm additionally supervises easy occurrence and conditional B/H/R magnitude,without tying an event probability to a mean-harm ratio.',
        'This adapts the old easy-hurdle idea to the full current bank,not a new hurdle method or exact old-critic continuation.',
        'The predicted event factor is not the realized future easy label. Unknown labels stay NaN and are never inference inputs.','',
        '|Arm|Fits|Updates|Initial anchor|Final anchor|Initial occurrence BCE|Final occurrence BCE|Initial conditional MSE|Final conditional MSE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        cells=[r for r in v['fits'] if r['arm']==arm];mean=lambda trace,key:sum(r[trace]['monitor'][key] for r in cells)/len(cells)
        values=[mean(t,k) for k in ('anchor_objective','occurrence_BCE','conditional_magnitude_MSE') for t in ('first_trace','final_trace')]
        lines.append(f'|{arm}|{len(cells)}|{sum(r["step"] for r in cells)}|'+'|'.join(cell(x) for x in values)+'|')
    lines+=['','These are fixed fitting-query monitors,not validation or checkpoint selection. Auxiliary components are only diagnostic in the marginal arm.','',
        '## Complete Policies','',
        '|Policy|Known ADE gain % [conditional95%CI]|FDE gain % (not estimable if incomplete)|Completion-lower utility pp|Supported record views|Known all/easy violations|Undefined selected easy|Worst known easy degradation %|Switch rate|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in run.evaluation.NAMES:
        r=s['safety'][name];x=a[name]
        lines.append(f'|{name}|{interval(s["accuracy"][name])}|{interval(s["FDE"][name])}|{interval(s["floor_lower"][name])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{cell(x["worst_known_easy_degradation_percent"])}|{cell(x["switch_rate"])}|')
    lines+=['','Floor abstention has no selected denominator. That is not evidence the floor causes harm or a positive learned-intervention result.','',
        '## Registered Contrasts','',
        '|Contrast|Scope|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|','|---|---|---:|---:|']
    for n,e in s['effects'].items():
        for mode,x in e.items():lines.append(f'|{n}|{mode}|{interval(x["known_pp"])}|{interval(x["lower_pp"])}|')
    lines+=['','Primary:identified-minus-marginal FULL completion-lower. Empty required common-count views are not_estimable,not safe0.',
        'Count matching is utility only. Paired completion bounds use unknown-row path disagreement,not subtraction of individual lower bounds.',
        'Uncertainty about missing outcomes is not measured hidden harm. Original recording risks remain required.','',
        '## Fitting-Cohort Diagnostic Scores','',
        '|Quantity|Marginal [conditional95%CI]|Identified [conditional95%CI]|Paired identified-minus-marginal|','|---|---:|---:|---:|']
    lines.append(f'|Easy occurrence Brier|{interval(s["easy_occurrence_Brier"]["marginal"])}|{interval(s["easy_occurrence_Brier"]["identified"])}|{interval(v["diagnostic_effects"]["easy_occurrence_Brier"])}|')
    for k,label in enumerate(('B','H','R','ER','EH')):
        lines.append(f'|{label} cost-scale normalized MSE|{interval(s["normalized_moment_MSE"]["marginal"][str(k)])}|{interval(s["normalized_moment_MSE"]["identified"][str(k)])}|{interval(v["diagnostic_effects"]["normalized_moment_MSE"][str(k)])}|')
    lines+=['','MSE/Brier are fitting-cohort diagnostics,not independent calibration or an actual-risk guarantee. Lower is better here.','',
        '|Arm|Pointwise blocked views|Singleton joint obstruction views|Rejected safe easy actions|Rejected beneficial safe easy actions|Admitted unsafe actions|','|---|---:|---:|---:|---:|---:|']
    for arm,r in s['guard_diagnostic'].items():lines.append(f'|{arm}|{r["pointwise_blocked_views"]}|{r["native_joint_singleton_proof_views"]}|{r["safe_easy_rejected"]}|{r["beneficial_safe_easy_rejected"]}|{r["known_unsafe_admitted"]}|')
    lines+=['','These repeat candidates,contexts and seeds. Labels diagnose error only after the causal choices are frozen;no oracle-filtered policy is deployed.','',
        '## Compute And Reproduction','']
    for label,r in (('Actual400-update pilot',v['pilot']),('Formal training and exact full refits',v['training']),('Complete72readout',v['complete']),('Separate numerical replay',v['replay'])):
        lines.append(f'{label}:PID{r["pid"]},{r["seconds"]:.6f}s,peakRSS{r["peak_RSS_bytes"]:,}B.')
    lines+=['','Native arm64CPU4/inter-op1/workers0. Atomic50-update checkpoints,heartbeat,explicit resume. No speed-based budget reduction.','',
        '## Fitting Gates','']+[f'- {k}: {"pass" if value else "fail"}' for k,value in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement"]}. Independent confirmation:false. Deployment changed:false.','',
        '![Conditional moment comparison and real fitting curves](figures/conditional_moment.svg)','',
        '## Limits','',
        'All149,510targets/74recordings/12localities/72groups remain.3,000locality resamples first average repeated context and cost-seed views.',
        'These SOURCE intervals are conditional development evidence. No three-forecaster,independent-selection/calibration/test or whole-pipeline OOF claim.',
        'The factorization targets known-label conditional costs,not unobserved outcome identification or a missing-at-random assumption.',
        'EuropeanSquares detector-silver/oblique local-image,obs8/pred12/stride12rawframes. Not SDDt+50,metric/seconds,human-gold,true3D,foundation,new dynamics or physical safety.',
        'Stage5C and SMC remain off.','']
    (p/'readout.md').write_text('\n'.join(lines));run.write(p/'supplementary_statistics.json',a);run.write(p/'diagnostic_effects.json',v['diagnostic_effects'])
    (p/'gates.md').write_text('# Conditional-Moment Fitting Gates\n\n'+'\n'.join(f'- {k}: {value}' for k,value in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement"]}. No independent confirmation/deployment. Stage5C/SMC off.\n')
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),complete_sha256=run.sha(p/'complete.json'),verification_sha256=run.sha(p/'verification.json'),
        new_real_fits=36,primary_updates=208320,exact_full_refit_updates=14336,fitting_advancement=s['fitting_advancement'],independent_confirmation=False,deployment_changed=False))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];fig,grid=plt.subplots(2,2,figsize=(11,8),layout='constrained');names=('frozen_tdef_guarded','marginal_guarded','identified_guarded')
    for i,n in enumerate(names):
        x=s['accuracy'][n];grid[0,0].bar(i,x['point'],color=('#555555','#326a95','#9a6a16')[i],width=.6)
        if x['CI95'] is not None:grid[0,0].vlines(i,*x['CI95'],color='black')
    grid[0,0].set_xticks(range(3),['Frozen guard','Marginal','Identified']);grid[0,0].set_ylabel('Known ADE gain over floor (%)')
    for i,key in enumerate(('known_pp','lower_pp')):
        x=s['effects']['identified_minus_marginal']['full'][key]
        if x['CI95'] is not None:grid[0,1].hlines(i,*x['CI95'],color='#326a95');grid[0,1].plot(x['point'],i,'o',color='#326a95')
    grid[0,1].set_yticks([0,1],['Known','Completion lower']);grid[0,1].set_ylim(-.3,1.3);grid[0,1].axvline(0,color='black',linewidth=.7)
    grid[0,1].set_xlabel('Identified minus marginal utility (pp)');progress=run.api.np.linspace(0,1,101)
    for arm,color in (('marginal','#326a95'),('identified','#9a6a16')):
        cells=[r for r in v['traces']['cells'] if r['arm']==arm]
        for axis,key in zip(grid[1],('anchor_monitor','occurrence_monitor')):
            curves=[run.api.np.interp(progress,[t['step']/r['budget'] for t in r['trace']],[t[key] for t in r['trace']]) for r in cells]
            axis.plot(progress,run.api.np.mean(curves,axis=0),color=color,label=arm)
            axis.set_xlabel('Fraction of each complete update budget');axis.set_title('Fixed fitting-query monitor; not validation');axis.legend()
    grid[1,0].set_ylabel('Same anchor objective');grid[1,1].set_ylabel('Easy occurrence BCE (measured in both arms)')
    for axis in grid.flat:axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Exposed SOURCE: coherent conditional costs, fixed ranking')
    (run.PUBLIC/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(run.PUBLIC/'figures/conditional_moment.svg');fig.savefig(run.PRIVATE/'conditional_moment.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    value=build();render(value);figure(value);print(json.dumps(dict(groups=72,fitting_advancement=value['summary']['fitting_advancement'],independent_confirmation=False)))
