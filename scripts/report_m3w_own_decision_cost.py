"""Public-scalar reconstruction of the complete fixed/own-decision cost experiment."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_own_decision_cost as run
from src.evaluation.m3w_metric_bundle import verify


def training():
    p=run.PUBLIC;freeze=json.loads((p/'training_freeze.json').read_text());reg=json.loads(run.REG.read_text())
    expected={(c,s,a) for c in reg['contexts'] for s in reg['seeds'] for a in run.api.ARMS};cells={}
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Public fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text());key=r['context'],r['seed'],r['arm']
        if key in cells:raise ValueError('Duplicate cost fit')
        if (r['registration_sha256']!=run.sha(run.REG) or r['step']!=r['budget'] or r['minimum_query_visits']<4
                or r['never_drawn_queries'] or r['parameters']!=31119 or r['trainable_parameters']!=31119
                or r['unknown_rows_with_supervised_labels']):raise ValueError('Complete matched real training required')
        counts=r['selected_query_counts']
        if sum(counts[k] for k in ('empty_selected_queries','unknown_selected_queries','complete_selected_queries'))!=16*r['step']:
            raise ValueError('Every sampled training query must retain its support status')
        if r['training_solver_calls']!=(16*r['step'] if r['arm']=='own_choice' else 0):raise ValueError('Current-choice native solver count changed')
        cells[key]=r
    if set(cells)!=expected or len(cells)!=36 or freeze['registration_sha256']!=run.sha(run.REG):raise ValueError('All36registered fits required')
    for context in reg['contexts']:
        for seed in reg['seeds']:
            a,b=(cells[context,seed,arm] for arm in run.api.ARMS)
            for k in ('input_hashes','draw_hash','budget','source_query_counts','source_completed_cycles','warm_checkpoint','unknown_rows_drawn'):
                if a[k]!=b[k]:raise ValueError('Paired training source/draw/budget changed')
    updates=sum(r['step'] for r in cells.values());refits=[]
    for arm in run.api.ARMS:
        x=json.loads((p/f'first_full_refit_{arm}.json').read_text())
        if not x['scientific_state_exact'] or not x['wall_time_excluded']:raise ValueError('Actual full-budget scientific refits required')
        refits.append(x['updates'])
    if updates!=208320 or freeze['primary_optimizer_updates']!=updates or sum(refits)!=14336 or freeze['exact_full_refit_updates']!=sum(refits):
        raise ValueError('Original complete-update budgets required')
    total={}
    for arm in run.api.ARMS:
        values=[r for r in cells.values() if r['arm']==arm]
        total[arm]=dict(fits=len(values),updates=sum(r['step'] for r in values),
            training_solver_calls=sum(r['training_solver_calls'] for r in values),
            training_solver_seconds=sum(r['training_solver_seconds'] for r in values),
            selected_query_counts={k:sum(r['selected_query_counts'][k] for r in values) for k in values[0]['selected_query_counts']},
            minimum_query_visits=min(r['minimum_query_visits'] for r in values),never_drawn_queries=0)
    return dict(freeze=freeze,totals=total)


def supplementary(rows):
    tails={};selected={};reasons={};nonoptimal={}
    for name in run.evaluation.NAMES:
        cells=[r['result']['recordings'][record][name] for r in rows for record in r['result']['recordings']]
        values=[100*(c['selected_known_easy_harm_mass']-c['selected_known_easy_benefit_mass'])/c['full_known_easy_reference_mass']
            for c in cells if c['full_known_easy_reference_mass']>0]
        tails[name]=dict(required_views=len(cells),known_defined=len(values),worst_known_easy_degradation_percent=max(values) if values else None,
            known_above2_percent=sum(x>2+1e-10 for x in values))
    for name in ('fixed_joint','own_joint'):
        cells=[r['result']['selected_cost_diagnostic'][name] for r in rows];n=sum(c['known_selected_rows'] for c in cells)
        v=[run.api.np.asarray(c['normalized_bias'])*c['known_selected_rows'] for c in cells if c['known_selected_rows']]
        selected[name]=dict(known_selected_row_views=n,pooled_normalized_bias=(sum(v)/n).tolist() if n else None,
            harm_underprediction_row_views=sum(c['harm_underprediction_rows'] for c in cells),
            easy_harm_underprediction_row_views=sum(c['easy_harm_underprediction_rows'] for c in cells))
    for r in rows:
        for name,v in r['result']['allocation']['reason_counts'].items():
            a=reasons.setdefault(name,{})
            for key,n in v.items():a[key]=a.get(key,0)+n
        for name,n in r['result']['allocation']['nonoptimal'].items():nonoptimal[name]=nonoptimal.get(name,0)+n
    return dict(easy_tails=tails,selected_cost_diagnostic=selected,
        query_views=sum(r['result']['allocation']['queries'] for r in rows),solver_calls=sum(r['result']['allocation']['solver_calls'] for r in rows),
        solver_reason_counts=reasons,nonoptimal_solver_queries=nonoptimal,repeated_views_not_independent_samples=True,
        selected_bias_is_SOURCE_description_not_independent_calibration=True)


def build():
    reg=json.loads(run.REG.read_text());cfg=json.loads(run.CONFIG.read_text());p=run.PUBLIC
    for path,digest in dict(reg['bindings'],**reg['dependencies']).items():
        if run.sha(ROOT/path)!=digest:raise ValueError('Registered model/loss/allocator changed')
    for name,digest in reg['parent_files'].items():
        if run.sha(run.previous.PUBLIC/name)!=digest:raise ValueError('Original complete projection study changed')
    run.prior_report.build();trained=training();bundle=verify(ROOT,p)
    complete=json.loads((p/'complete.json').read_text());replay=json.loads((p/'verification.json').read_text())
    if (complete['training_freeze_sha256']!=run.sha(p/'training_freeze.json') or complete['summary_sha256']!=run.sha(p/'summary.json')
            or replay['summary_sha256']!=run.sha(p/'summary.json') or replay['complete_sha256']!=run.sha(p/'complete.json')
            or not replay['all72_new_cost_inference_and_frozen_action_scores_exact']
            or not replay['original_full_and_record_control_exact'] or replay['optimization_replayed']):raise ValueError('Complete readout and actual numerical replay required')
    groups=run.aggregate_groups(run.NAME);old=run.aggregate_groups(run.previous.NAME)
    rows=[groups[Path(ref['path']).name] for ref in complete['groups']];records=set()
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json';prior=old[key]
        if (row['registration_sha256']!=run.sha(run.REG) or row['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
                or row['parent_complete_sha256']!=run.sha(run.previous.PUBLIC/'complete.json') or row['ids_sha256']!=prior['ids_sha256']
                or row['action_sha256']['frozen_projected_joint']!=prior['action_sha256']['learned_joint']
                or row['result']['policies']['frozen_projected_joint']!=prior['result']['policies']['learned_joint']):raise ValueError('Original target/action/full-control lineage changed')
        for rec,v in row['result']['recordings'].items():
            if v['frozen_projected_joint']!=prior['result']['recordings'][rec]['learned_joint']:raise ValueError('Original recording control changed')
        records.update((row['source'],r) for r in row['result']['recordings'])
    rebuilt=run.evaluation.summarize(rows,reg['contexts'],cfg);rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if rebuilt!=json.loads((p/'summary.json').read_text()) or len(records)!=74:raise ValueError('Full72canonical summary required')
    return dict(summary=rebuilt,training=trained,complete=complete,replay=replay,bundle=bundle,auxiliary=supplementary(rows),
        pilot=json.loads((p/'pilot.json').read_text()))


cell,interval=run.prior_report.cell,run.prior_report.interval


def render(v):
    s=v['summary'];p=run.PUBLIC;a=v['auxiliary'];t=v['training'];main=s['effects']['own_minus_fixed']['full']
    run.write(p/'supplementary_statistics.json',a);run.write(p/'training_statistics.json',t['totals'])
    lines=['# Fixed Versus Own-Decision Coherent Cost Learning','','## Material Passport','',
        'fresh_run:36real Torch cost fits,complete72-group allocation and separate inference/frozen-action verification.',
        'cached_verified:seven forecasts,462causal fields,normalizers,labels,warm weights and original projected policy.',
        'not_run:forecaster/JEPA/Transformer/residual training,independent selection/calibration/confirmation and deployment.',
        'All results are exposed SOURCE fitting/resubstitution,not independent scientific confirmation.','',
        '## One Controlled Factor','',
        'Both arms have31,119active/total parameters,the same warm state,differentiable reference-sharing/benefit-cap decoder,draws and full budgets.',
        'Only the query-risk residual target changes:fixed causal initial choices versus detached current native joint choices.',
        'All known legal pointwise supervision remains. Empty/unknown-selected query targets are not supervised zero.',
        'The network context input stays fixed in both arms. No gradient through the integer solver or published consistency theorem is claimed.','',
        '|Arm|Fits|Updates|Gradient query solver calls|Gradient solver seconds|Complete/unknown/empty selected query views|',
        '|---|---:|---:|---:|---:|---:|']
    for name,x in t['totals'].items():
        c=x['selected_query_counts'];lines.append(f'|{name}|{x["fits"]}|{x["updates"]}|{x["training_solver_calls"]}|{cell(x["training_solver_seconds"])}|{c["complete_selected_queries"]}/{c["unknown_selected_queries"]}/{c["empty_selected_queries"]}|')
    lines+=['','The two first-context full refits add14,336updates and match all scientific state exactly,excluding only measured wall times.',
        'Every query receives at least four gradient sampling visits;none is undrawn. Unknown rows still enter causal context and choices.','',
        '## Registered Primary And Secondary Contrasts','',
        'Primary:own joint minus fixed joint FULL completion-lower utility. Frozen-parent and own-floor lowerCI and all recording risks must also pass.',
        'Same2%contract,full cohort and source roles;no seed/checkpoint/subset replaces the primary.','',
        '|Contrast|Scope|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|Completion-upper pp [conditional95%CI]|',
        '|---|---|---:|---:|---:|']
    for name,effects in s['effects'].items():
        for mode,x in effects.items():lines.append(f'|{name}|{mode}|{interval(x["known_pp"])}|{interval(x["lower_pp"])}|{interval(x["upper_pp"])}|')
    lines+=['','Post-hoc trimmed controls are utility-only;required empty views remain not_estimable,not safe0.',
        'The native joint allocator preserves its own independent intervention count;common-count cross-arm controls are separate.','',
        '|Policy|Known ADE gain over floor (%)|Full-roster FDE|Own completion-lower pp|Risk-supported recording views|Known all/easy risk violations|Undefined selected easy|Selected/unknown views|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in run.evaluation.NAMES:
        r=s['safety'][name];lines.append(f'|{name}|{interval(s["accuracy"][name])}|{interval(s["FDE"][name])}|{interval(s["floor_lower"][name])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{r["selected"]}/{r["selected_unknown"]}|')
    lines+=['','## Cost Error And Chosen-Action Bias','',
        'Normalized SOURCE fitting MSE,not independent calibration. Moments:B,H,R,R_easy,H_easy.','',
        '|Moment|Fixed choices|Own choices|','|---|---:|---:|']
    for k in range(5):lines.append(f'|{k}|{interval(s["cost_MSE"]["fixed_choice"][str(k)])}|{interval(s["cost_MSE"]["own_choice"][str(k)])}|')
    lines+=['','|Policy|Known selected views|B bias|H bias|R bias|R_easy bias|H_easy bias|','|---|---:|---:|---:|---:|---:|---:|']
    for name,x in a['selected_cost_diagnostic'].items():
        b=x['pooled_normalized_bias'];lines.append(f'|{name}|{x["known_selected_row_views"]}|'+
            '|'.join(cell(None if b is None else b[k]) for k in range(5))+'|')
    lines+=['','Bias pools repeated SOURCE views after original causal normalization. Negative H bias means underprediction,not independent calibration.',
        'All projected coherence checks are public. Algebraic coherence is structural,not evidence of trajectory accuracy or risk control.','',
        '## Easy Tails And Hard Cases','',
        '|Policy|Known easy defined|Worst known net easy degradation (%)|Recording views above2%|',
        '|---|---:|---:|---:|']
    for name,x in a['easy_tails'].items():lines.append(f'|{name}|{x["known_defined"]}/{x["required_views"]}|{cell(x["worst_known_easy_degradation_percent"])}|{x["known_above2_percent"]}|')
    lines+=['','Known harm,missing-outcome uncertainty,and net easy degradation are different quantities. Both original all/easy risk requirements remain.',
        'Pooled known-label subsets below repeat source/context/cost-seed views;they are not independent samples.','',
        '|Subset|Frozen projected policy|Fixed joint|Own joint|','|---|---:|---:|---:|']
    for subset in ('hard','easy'):
        x=s['subsets'][subset]['policies'];lines.append('|'+subset+'|'+'|'.join(cell(x[n]['ADE_gain_percent']) for n in ('frozen_projected_joint','fixed_joint','own_joint'))+'|')
    lines+=['','## Compute And Verification','']
    for name,x in (('Real400-update pilot',v['pilot']),('Formal training including full refits',t['freeze']),('Complete policy readout',v['complete']),('Separate inference/frozen-action verification',v['replay'])):
        lines.append(f'{name}:PID{x["pid"]},{x["seconds"]:.6f}s,peakRSS{x["peak_RSS_bytes"]}B.')
    lines+=['',f'Readout:{a["query_views"]}repeated query views,{a["solver_calls"]}solver calls.',
        'Actual training interruption/resume and18-query allocation recovery passed. Full refits exclude wall time,not loss/choice/model state.',
        'Final verification re-infers every new cost and scores frozen choices,not timed MILP reoptimization.',
        'Original full and recording scores remain exact. Failed/invalid solves keep the declared feasible independent parent.',
        'Native arm64CPU4/interop1/workers0;checkpoint50updates,readout250queries,live heartbeat,explicit resume,no speed-based reduction.','',
        '## Fitting Gates','']+[f'- {k}: {"pass" if x else "fail"}' for k,x in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement"]}. Independent confirmation:false. Deployment changed:false. Stage5C:false. SMC:false.','',
        '![Own-decision SOURCE results](figures/own_decision_readout.svg)','',
        '## Evidence Limits','',
        'All149,510targets,74recordings,72groups and12exposed localities remain.3,000paired locality bootstrap draws average repeated context/cost-seed views first.',
        'The three cost seeds share forecaster43. SOURCE fitting intervals are not three independently retrained forecast pipelines or independent evaluation.',
        'Unknown endpoints keep full-roster FDE not_estimable;no earlier endpoint substitution or fabricated future interior labels.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'No new world dynamics is established by cost-head training. Public scalar evidence does not replace independent role-compliant experiments.','']
    (p/'readout.md').write_text('\n'.join(lines))
    (p/'gates.md').write_text('# Own-Decision Cost Fitting Gates\n\n'+'\n'.join(f'- {k}: {x}' for k,x in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement"]}. Independent confirmation:false. Deployment:false. Stage5C:false. SMC:false.\n')
    with (p/'locality_effects.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['locality','known_pp','completion_lower_pp','completion_upper_pp'])
        for source in sorted(main['known_pp'].get('by_locality',{})):writer.writerow([source]+[main[k]['by_locality'][source] for k in ('known_pp','lower_pp','upper_pp')])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),training_freeze_sha256=run.sha(p/'training_freeze.json'),
        complete_sha256=run.sha(p/'complete.json'),verification_sha256=run.sha(p/'verification.json'),archive_sha256=v['bundle']['archive_sha256'],
        real_cost_fits=36,primary_optimizer_updates=208320,exact_refit_updates=14336,new_trajectory_training=False,
        fitting_advancement=s['fitting_advancement'],independent_confirmation=False,deployment_changed=False,submission_ready=False))


def interpretation(v):
    s=v['summary'];p=run.PUBLIC;main=s['effects']['own_minus_fixed']['full'];r=s['safety']['own_joint'];a=v['auxiliary']
    lines=['# Own-Decision Cost Learning: Interpretation','',
        f'Registered own-minus-fixed known effect:{interval(main["known_pp"])}pp.',
        f'Registered full completion-lower effect:{interval(main["lower_pp"])}pp.',
        f'Own-minus-frozen full lower:{interval(s["effects"]["own_minus_frozen"]["full"]["lower_pp"])}pp.',
        f'Original recording-risk support:{r["supported_record_views"]}/{r["required_record_views"]};undefined selected easy:{r["undefined_easy_record_views"]}.',
        f'Worst known easy degradation:{cell(a["easy_tails"]["own_joint"]["worst_known_easy_degradation_percent"])}%.','',
        'Both arms change training through the same coherent decoder. Their paired primary isolates the choice set used in the aggregate risk residual,',
        'not the entire effect of coherent learning versus the preceding untrained postprocessor. Fixed model context,forecasts and all source roles stay frozen.',
        'Pointwise labels still cover all known legal actions. Missing selected outcomes are not supervised zero;empty choices receive no aggregate target.',
        'Training support counts expose whether this aggregate objective is actually active,not an oracle or a deployment metric.',
        'A favorable component contrast cannot rescue a failed primary or any original recording-risk requirement.',
        'A negative result would reject this stop-gradient residual objective\'s sufficiency on these SOURCE outputs,not all decision-focused training.',
        'Selected cost bias is descriptive. It does not identify a unique causal failure mechanism or certify conditional risks.',
        'If the objective remains inadequate,inspect selected-label support and harm error before changing the loss;do not silently reduce the cohort or risk contract.',
        'No independent data are opened to rescue a SOURCE fit. This is not proof of neural trajectory,JEPA,scene or interaction contribution.','']
    (p/'failure_analysis.md').write_text('\n'.join(lines))
    (p/'model_data_card.md').write_text('# Cost Model And Data Card\n\n'
        '36real warm-started Torch cost heads,31,119active parameters each. No new trajectory or latent generator.\n'
        'Shared differentiable decoder:causal available-candidate R/R_easy mean,B capped at R,H/H_easy retained.\n'
        'Input:462frozen causal fields and original causal cost/envelope/initial-choice context. Future labels enter losses/evaluation only.\n'
        'Two objectives differ only in fixed versus detached current native choices used by the query-risk residual.\n'
        'Training targets are pointwise known-action costs and complete nonempty selected-query residuals. Unknown targets stay unknown.\n'
        'EuropeanSquares exposed SOURCE,149,510targets,74recordings,12localities,obs8/pred12/rawstride12.\n'
        'Oblique detector-silver/local-image,not metric,seconds,human gold,true3D or foundation.\n'
        'Three cost seeds share one forecaster. No independent selection/calibration/confirmation or whole-pipeline OOF claim.\n'
        'Every original2%risk and undefined-support rule stays unchanged. Predicted feasibility is not physical safety.\n'
        'Raw data,caches,features,weights and third-party media remain outsideGit. Public scalar evidence is reproducible separately.\n')
    (p/'publication_gap.md').write_text('# Remaining Submission Evidence\n\n'
        'Not yet a submission candidate. Cost regression and learning constraints are established areas,not novel from this combination alone.\n'
        'Still required:useful complete-policy effects under original risks,independent role-compliant full-pipeline fitting and confirmation,\n'
        'matched strong published deferral/constraint-learning baselines,three forecast seeds where claimed,adequate outcome support,\n'
        'scene/interaction contribution and complete anonymous paper/supplement/reproduction materials.\n'
        'This SOURCE control cannot alone authorize deployment,Stage5C or SMC.\n')


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,(name,color) in enumerate(zip(('frozen_projected_joint','fixed_joint','own_joint'),('#326a95','#5b8b70','#b85042'))):
        x=s['accuracy'][name];axes[0].bar(i,x['point'],color=color,width=.6)
        if x['CI95'] is not None:axes[0].vlines(i,*x['CI95'],color='#222222')
    axes[0].set_xticks(range(3),['Frozen projection','Fixed-choice fit','Own-choice fit']);axes[0].set_ylabel('Known ADE gain over floor (%)')
    axes[0].set_title('Same forecasts, warm weights and active capacity',fontsize=11)
    for i,key in enumerate(('known_pp','lower_pp')):
        x=s['effects']['own_minus_fixed']['full'][key]
        if x['CI95'] is not None:axes[1].hlines(i,*x['CI95'],color='#326a95');axes[1].plot(x['point'],i,'o',color='#326a95')
    axes[1].axvline(0,color='#333333',lw=.8);axes[1].set_yticks([0,1],['Known labels','Completion lower'])
    axes[1].set_xlabel('Own minus fixed joint utility (pp)');axes[1].set_title('Paired conditional 95% intervals',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE fitting only: own-decision cost learning',fontsize=12)
    dest=run.PUBLIC/'figures';dest.mkdir(parents=True,exist_ok=True);svg=dest/'own_decision_readout.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    run.PRIVATE.mkdir(parents=True,exist_ok=True);fig.savefig(run.PRIVATE/'own_decision_readout.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    v=build();render(v);interpretation(v);figure(v)
    print(json.dumps(dict(groups=v['summary']['groups'],real_cost_fits=36,primary=v['summary']['effects']['own_minus_fixed']['full']['lower_pp'],
        fitting_advancement=v['summary']['fitting_advancement'],independent_confirmation=False)))
