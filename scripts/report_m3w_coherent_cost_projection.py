"""Rebuild the complete projection control from public scalar evidence only."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_coherent_cost_projection as run
from scripts import report_m3w_reference_cost_refit as preceding
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def supplementary(rows):
    tails={};pairs={};support={};reasons={};nonoptimal={}
    n_pairs=sum(r['result']['pair_proxy']['pair_requested_step_views'] for r in rows)
    for a in run.evaluation.NAMES:
        values=[v[a] for r in rows for v in r['result']['recordings'].values()]
        known=[100*(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/
            v['full_known_easy_reference_mass'] for v in values if v['full_known_easy_reference_mass']>0]
        upper=[100*v['easy_degradation_upper'] for v in values if v['easy_degradation_upper'] is not None]
        tails[a]=dict(required_record_views=len(values),known_defined=len(known),
            worst_known_easy_degradation_percent=max(known) if known else None,
            known_above2_percent=sum(v>2+1e-10 for v in known),
            completion_upper_defined=len(upper),worst_completion_upper_percent=max(upper) if upper else None)
        weighted=[]
        for r in rows:
            x=r['result']['pair_proxy'];n=x['pair_requested_step_views']
            if n:
                value=x['mean_positive_distance_contraction_over_frozen_scale'][a]
                if value is None:raise ValueError('Positive proxy support requires an observed value')
                weighted.append(n*value)
        p95=[v[a]['p95'] for r in rows for v in r['result']['record_tails'].values() if v[a]['p95'] is not None]
        pairs[a]=dict(mean_pair_contraction=sum(weighted)/n_pairs if n_pairs else None,
            maximum_recording_p95=max(p95) if p95 else None)
    for arm in run.api.ARMS:
        result={}
        for row in rows:
            for name,metrics in row['result']['pointwise_diagnostic'][arm]['supports'].items():
                out=result.setdefault(name,{k:0 for k in ('allowed_action_occurrences','supported_record_views','positive_record_views')})
                for k in out:out[k]+=metrics[k]
        support[arm]=result
    for r in rows:
        for a,n in r['result']['allocation']['nonoptimal'].items():nonoptimal[a]=nonoptimal.get(a,0)+n
        for a,values in r['result']['allocation']['reason_counts'].items():
            out=reasons.setdefault(a,{})
            for k,n in values.items():out[k]=out.get(k,0)+n
    selected={}
    for a in ('raw_locked_joint','raw_learned_joint','locked_joint','learned_joint'):
        cells=[r['result']['selected_cost_diagnostic'][a] for r in rows];n=sum(x['known_selected_rows'] for x in cells)
        values=[]
        for x in cells:
            if x['known_selected_rows']:
                if x['normalized_bias'] is None:raise ValueError('Nonempty selected labels require a bias vector')
                values.append(run.api.np.asarray(x['normalized_bias'])*x['known_selected_rows'])
        selected[a]=dict(known_selected_row_views=n,
            pooled_normalized_bias=(sum(values)/n).tolist() if n else None,
            selected_harm_underprediction_row_views=sum(x['selected_harm_underprediction_rows'] for x in cells),
            selected_easy_harm_underprediction_row_views=sum(x['selected_easy_harm_underprediction_rows'] for x in cells))
    return dict(easy_tails=tails,pair_and_tail=pairs,pointwise_label_aware_support=support,
        query_views=sum(r['result']['allocation']['queries'] for r in rows),
        solver_calls=sum(r['result']['allocation']['solver_calls'] for r in rows),
        nonoptimal_solver_queries=nonoptimal,solver_reason_counts=reasons,selected_cost_diagnostic=selected,
        repeated_pair_requested_step_views=n_pairs,repeated_views_not_independent_samples=True,
        selected_bias_is_pooled_SOURCE_description_not_calibration=True,physical_safety_certificate=False)


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text());cfg=json.loads(run.CONFIG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Registered projection definition changed')
    for name,digest in reg['parent_files'].items():
        if run.sha(run.parent.PUBLIC/name)!=digest:raise ValueError('Original complete cost study changed')
    previous=preceding.build();bundle=verify(ROOT,p)
    complete=json.loads((p/'complete.json').read_text());replay=json.loads((p/'verification.json').read_text())
    if (complete['summary_sha256']!=run.sha(p/'summary.json') or replay['summary_sha256']!=run.sha(p/'summary.json')
            or replay['complete_sha256']!=run.sha(p/'complete.json')
            or not replay['all72_projection_inference_and_frozen_action_scores_exact']
            or not replay['original_full_and_record_controls_exact'] or replay['optimization_replayed']):
        raise ValueError('Complete readout and separate full original-score verification required')
    groups=aggregate_groups(run.NAME);old=aggregate_groups(run.parent.NAME)
    rows=[groups[Path(r['path']).name] for r in complete['groups']];records=set()
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json';raw=old[key]
        if (row['registration_sha256']!=run.sha(run.REG) or row['parent_complete_sha256']!=run.sha(run.parent.PUBLIC/'complete.json')
                or row['ids_sha256']!=raw['ids_sha256'] or row['raw_prediction_sha256']!=raw['prediction_sha256'] or row['new_training']):
            raise ValueError('Complete original rows,predictions and unchanged training roles required')
        for prefix,arm in zip(('locked','learned'),run.api.ARMS):
            name=prefix+'_joint';target='raw_'+name
            if (row['action_sha256'][target]!=raw['action_sha256'][name]
                    or row['result']['policies'][target]!=raw['result']['policies'][name]
                    or row['result']['cost_MSE']['raw_'+prefix]!=raw['result']['calibration_MSE'][arm]):
                raise ValueError('Original native policy or cost score changed')
            for rec,values in row['result']['recordings'].items():
                if values[target]!=raw['result']['recordings'][rec][name]:raise ValueError('Original record control changed')
        records.update((row['source'],r) for r in row['result']['recordings'])
    s=json.loads((p/'summary.json').read_text());rebuilt=run.evaluation.summarize(rows,reg['contexts'],cfg)
    rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if s!=rebuilt or len(records)!=74:raise ValueError('All72 canonical groups and74recordings required')
    return dict(summary=s,complete=complete,replay=replay,bundle=bundle,auxiliary=supplementary(rows),
        old_training=previous['training'],pilot=json.loads((p/'pilot.json').read_text()))


cell=preceding.cell
interval=preceding.interval


def render(v):
    p=run.PUBLIC;s=v['summary'];aux=v['auxiliary'];main=s['effects']['learned_projection']['full']
    run.write(p/'supplementary_statistics.json',aux)
    lines=['# Zero-Step Coherent-Cost Control: Complete SOURCE Readout','','## Material Passport','',
        'fresh_run:causal cost projection,complete72-group allocation/readout,and separate full inference/frozen-action verification.',
        'cached_verified:all36previously trained cost endpoints,seven forecasts,462causal fields,normalizers,labels and both original full-joint policies.',
        'not_run:new cost/forecaster/JEPA/Transformer/residual training,independent calibration/confirmation or deployment.',
        'All present outcomes are exposed SOURCE fitting/resubstitution,not independent test evidence.','',
        '## Controlled Change','',
        'Equal-average predicted R/R_easy over causally available candidates per agent;cap B at shared R. H/H_easy and all trajectories stay unchanged.',
        'No target,label availability,future endpoint,test goal,manual reference maximum,new action or no-op anchor enters projection.',
        'The sequential reference/benefit projection is not the closest joint projection of all five moments and is not new neural training.',
        'Reference MSE can fall by averaging alone. Only actual policy effects and recording risk address the research question.','',
        '## Registered Primary And Controls','',
        'Primary:projected learned joint minus original unprojected learned joint FULL completion-lower ADE utility.',
        'Locked projection and joint-minus-independent are secondary,not replacement primaries. Original2%requirements are unchanged.','',
        '|Contrast|Scope|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|Completion-upper pp [conditional95%CI]|',
        '|---|---|---:|---:|---:|']
    for name,effects in s['effects'].items():
        for mode,e in effects.items():lines.append(f'|{name}|{mode}|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    penalty=None if any(main[k]['point'] is None for k in ('known_pp','lower_pp')) else main['known_pp']['point']-main['lower_pp']['point']
    lines+=['',f'Paired known-minus-lower uncertainty penalty:{cell(penalty)}pp. Not measured harm.',
        'Original-versus-projected matched rows are trimmed per query by each own utility. These controls are utility-only,not safety policies.',
        'The allocator also retains separate projected locked/learned common-count controls;those are a different secondary comparison.',
        'Empty matched comparisons stay not_estimable. Never turn empty selected easy support into a safety pass.','',
        '|Policy|Known ADE gain over floor (%)|Known FDE gain (%)|Own completion-lower pp|Record-risk views supported|Known all/easy risk violations|Undefined selected easy|Selected/unknown views|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        r=s['safety'][a];lines.append(f'|{a}|{interval(s["accuracy"][a])}|{interval(s["FDE"][a])}|{interval(s["floor_lower"][a])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{r["selected"]}/{r["selected_unknown"]}|')
    lines+=['','## Cost Coherence','','Counts repeat source/context/cost-seed views. Excess tolerance1e-12in frozen causal-scale units.','',
        '|Costs|Legal action views|B>R|Negative implied ADE|R differs across candidates|R_easy differs across candidates|',
        '|---|---:|---:|---:|---:|---:|']
    for a,x in s['coherence'].items():
        lines.append(f'|{a}|{x["legal_action_views"]}|{x["violations"]["benefit_exceeds_reference"]}|{x["violations"]["negative_implied_candidate_ADE"]}|{x["reference_spread"]["reference"]}|{x["reference_spread"]["easy_reference"]}|')
    lines+=['','Zero contradictions are structural properties of this postprocessor,not proof of risk calibration or trajectory quality.',
        'Negative implied ADE describes predicted cost moments,not negative measured trajectory errors.','',
        '## Five Cost Errors','','Normalized SOURCE fitting MSE. Columns are B,H,R,R_easy,H_easy;not independent calibration.','',
        '|Moment|Raw locked|Projected locked|Raw learned|Projected learned|','|---|---:|---:|---:|---:|']
    for k in range(5):lines.append('|'+str(k)+'|'+'|'.join(interval(s['cost_MSE'][a][str(k)]) for a in ('raw_locked','projected_locked','raw_learned','projected_learned'))+'|')
    lines+=['','H/H_easy predictions remain identical;selected bias can nevertheless change because a different action is chosen.',
        'Per-group selected cost sums/bias are in the scalar archive. Supplementary pooled normalized bias is descriptive over repeated views,not a calibration guarantee.','',
        '## Easy Tails','','|Policy|Known easy defined|Worst known net degradation (%)|Views above2%|Worst completion-upper bound (%)|',
        '|---|---:|---:|---:|---:|']
    for a,x in aux['easy_tails'].items():lines.append(f'|{a}|{x["known_defined"]}/{x["required_record_views"]}|{cell(x["worst_known_easy_degradation_percent"])}|{x["known_above2_percent"]}|{cell(x["worst_completion_upper_percent"])}|')
    lines+=['','Completion-upper values bound missing outcomes;they are not observed catastrophic errors.',
        'Net easy degradation and selected positive-harm/reference risk have different numerators/denominators;both original requirements remain.','',
        '|Known-label subset|Raw learned joint|Projected learned joint|','|---|---:|---:|']
    for subset in ('hard','easy'):
        x=s['subsets'][subset]['policies'];lines.append('|'+subset+'|'+'|'.join(cell(x[a]['ADE_gain_percent']) for a in ('raw_learned_joint','learned_joint'))+'|')
    lines+=['','## Compute And Reproduction','',
        f'Actual resource/recovery probe PID{v["pilot"]["pid"]}:{v["pilot"]["seconds"]:.6f}s,peakRSS{v["pilot"]["peak_RSS_bytes"]}B.',
        f'Full readout PID{v["complete"]["pid"]}:{v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B.',
        f'Separate verification PID{v["replay"]["pid"]}:{v["replay"]["seconds"]:.6f}s,peakRSS{v["replay"]["peak_RSS_bytes"]}B.',
        f'Complete allocation:{aux["query_views"]}repeated query views,{aux["solver_calls"]}solver calls. No new optimizer updates.',
        'The actual18-query interruption/resume check is exact. Final verification re-infers costs and scores frozen actions,not timed MILP reoptimization.',
        'Every original full/record score and raw cost error is exact. Original cost weights and forecasts are unchanged.',
        'Nonoptimal/invalid solves keep the declared feasible independent parent. Solver reason counts are public.',
        'Native arm64CPU4/interop1/workers0,query checkpoints250,heartbeat and explicit resume. No speed-based reduction.','',
        '## Interaction And Tail Diagnostics','','Pair contraction is over query targets only,not all visible agents or a collision certificate.','',
        '|Policy|Mean pair contraction/frozen scale|Maximum record p95 ADE|','|---|---:|---:|']
    for a in ('raw_learned_joint','learned_joint','raw_locked_joint','locked_joint','floor'):
        x=aux['pair_and_tail'][a];lines.append(f'|{a}|{cell(x["mean_pair_contraction"])}|{cell(x["maximum_recording_p95"])}|')
    lines+=['','## Fitting Gates','']+[f'- {name}: {"pass" if value else "fail"}' for name,value in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement"]}. Deployment and independent roles unchanged.','',
        '![Coherent-cost SOURCE results](figures/coherent_cost_readout.svg)','',
        '## Evidence Limits','',
        'All149,510unique targets,74recordings,72groups and12exposed localities remain.3,000paired locality bootstrap draws average repeated contexts/cost seeds first.',
        'These are nominal SOURCE fitting intervals. Cost seeds17/29/43share forecaster43,not three whole forecast pipelines.',
        'Overlapping windows are not independent samples. Full-roster FDE stays not_estimable without requested endpoints;no earlier substitution or fabricated interior labels.',
        'Label-aware pointwise support is diagnostic,not model utility,and its failure alone does not prove query pooling infeasible.',
        'EuropeanSquares oblique-camera detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Stage5C and SMC remain disabled. This experiment alone cannot establish world dynamics,independent generalization or CCF-A submission readiness.','']
    (p/'readout.md').write_text('\n'.join(lines))
    (p/'gates.md').write_text('# Coherent-Cost Fitting Gates\n\n'+'\n'.join(f'- {k}: {x}' for k,x in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement"]}. New training:false. Independent confirmation:false. Deployment:false. Stage5C:false. SMC:false.\n')
    with (p/'locality_effects.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['locality','known_pp','completion_lower_pp','completion_upper_pp'])
        for source in sorted(main['known_pp'].get('by_locality',{})):
            writer.writerow([source]+[main[k]['by_locality'][source] for k in ('known_pp','lower_pp','upper_pp')])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'),archive_sha256=v['bundle']['archive_sha256'],
        frozen_models_cached_verified=36,new_training=False,fitting_advancement=s['fitting_advancement'],
        independent_confirmation=False,deployment_changed=False,submission_ready=False))


def interpretation(v):
    p=run.PUBLIC;s=v['summary'];primary=s['effects']['learned_projection']['full'];risk=s['safety']['learned_joint'];tail=v['auxiliary']['easy_tails']['learned_joint']
    text=['# Coherent Projection: Measured Interpretation','',
        'This follows the complete original roster and registered primary. No seed,threshold or subgroup replaces the primary.','',
        f'Primary known:{interval(primary["known_pp"])}pp;completion-lower:{interval(primary["lower_pp"])}pp.',
        f'Matched utility-only lower:{interval(s["effects"]["learned_projection"]["matched"]["lower_pp"])}pp.',
        f'Full recording-risk support:{risk["supported_record_views"]}/{risk["required_record_views"]};undefined selected easy:{risk["undefined_easy_record_views"]}.',
        f'Worst known easy degradation:{cell(tail["worst_known_easy_degradation_percent"])}%;views above2%:{tail["known_above2_percent"]}.','',
        '## What This Test Can Establish','',
        'It directly removes candidate-dependent references and B>R in the supplied costs without new fitting.',
        'The paired complete-policy contrast estimates the effect of this sequential deterministic intervention on these frozen SOURCE outputs.',
        'It does not separate reference sharing from benefit clipping,and does not prove that either individual defect causes the full previous failure.',
        'A failed frozen-output projection would reject sufficiency of this particular transform,not every coherently trained decoder.',
        'A favorable locked-arm or known-only result cannot rescue a failed learned-arm completion-lower primary or failed original risk contract.',
        'A reduction in reference MSE is partly mathematical averaging,not independent predictive evidence. H/H_easy are unchanged.',
        'Coherence is necessary for the stated cost interpretation,but not sufficient for accurate conditional risk or safe choices.',
        'Missing-outcome uncertainty is not observed damage. Actual known positive-harm violations and net easy degradation are reported separately.','',
        '## Next Evidence-Producing Action','',
        'If the primary and original risks pass,retain this frozen component result and test whether coherent decoder learning adds anything beyond the zero-step control.',
        'If they fail,do not simply add epochs to reference MSE. Inspect selected harm underprediction,record-level support and outcome coverage;',
        'then test a matched decision-aware/record-risk cost objective with a structurally coherent decoder,or repair inadequate labels when required.',
        'No independent data are opened to rescue a SOURCE fit. A zero-step projection is not neural world-dynamics learning.',
        'The original2%risk/undefined-support rules stay in force. Stage5C and SMC remain disabled.','']
    (p/'failure_analysis.md').write_text('\n'.join(text))
    (p/'publication_gap.md').write_text('# Remaining Publication Evidence\n\n'
        'This control is not a standalone novel method claim. Existing work already studies regression deferral among fixed experts.\n'
        'Still required:useful complete-policy method effects,all original recording-risk requirements,genuinely independent model selection/calibration/confirmation,\n'
        'role-compliant full-pipeline fitting,three forecast seeds where claimed,matched strong deferral baselines,scene/interaction contribution,adequate future-label support,\n'
        'and a complete anonymous paper/supplement/reproduction package. SOURCE fitting,coherence and engineering checks cannot substitute for these.\n'
        'No deployment,Stage5C or SMC promotion. Not yet a CCF-A submission candidate.\n')
    (p/'model_data_card.md').write_text('# Model And Data Card\n\n'
        'No new neural model or trajectory generator.36cached_verified cost endpoints and a fixed seven-forecast bank.\n'
        'Postprocessor:equal-average causal available-candidate references per agent,cap B at R,preserve H/H_easy and unavailable candidates.\n'
        'Inputs are predicted moments and causal availability only;no labels,future endpoints,central velocity,test goals or label-known masks.\n'
        'Data:EuropeanSquares exposed SOURCE,149,510unique targets,74recordings,12localities,obs8/pred12/rawstride12.\n'
        'Local-image oblique-camera detector-silver,not metric,seconds,human gold,true3D or foundation.\n'
        'Unknown labels remain unknown;missing requested endpoints/interiors are not repaired by bounds. Three cost seeds share forecaster43.\n'
        'Every original2%risk requirement is unchanged;predicted budget feasibility is not empirical or physical safety.\n'
        'Projection alone cannot identify conditional harm or establish dynamics/representation lift. Independent roles remain unopened.\n'
        'Raw data,caches,features and weights are not committed. Public scalar hashes,controls and reproduction commands are provided.\n')


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    names=('raw_locked_joint','locked_joint','raw_learned_joint','learned_joint');colors=('#326a95','#5b8b70','#9a6a16','#b85042')
    for i,(a,color) in enumerate(zip(names,colors)):
        z=s['accuracy'][a];ci=z['CI95'];axes[0].bar(i,z['point'],color=color,width=.6)
        if ci is not None:axes[0].errorbar(i,z['point'],yerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='none',ecolor='#222222',capsize=4)
    axes[0].set_xticks(range(4),['Raw locked','Projected\nlocked','Raw learned','Projected\nlearned'])
    axes[0].set_ylabel('Known ADE gain over floor (%)');axes[0].set_title('Fixed costs and forecasts; no training',fontsize=11)
    for i,k in enumerate(('known_pp','lower_pp')):
        z=s['effects']['learned_projection']['full'][k];ci=z['CI95']
        if ci is not None:axes[1].errorbar(z['point'],i,xerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='o',color=colors[i],capsize=5)
    axes[1].axvline(0,color='#333333',lw=.8);axes[1].set_yticks([0,1],['Known labels','Completion lower'])
    axes[1].set_xlabel('Projected minus raw learned joint utility (pp)');axes[1].set_title('Paired conditional 95% intervals',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE fitting only: causal coherence is not risk calibration',fontsize=12)
    dest=run.PUBLIC/'figures';dest.mkdir(exist_ok=True);svg=dest/'coherent_cost_readout.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    run.PRIVATE.mkdir(parents=True,exist_ok=True);fig.savefig(run.PRIVATE/'coherent_cost_readout.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    value=build();render(value);interpretation(value);figure(value)
    print(json.dumps(dict(groups=value['summary']['groups'],primary=value['summary']['effects']['learned_projection']['full']['lower_pp'],
        fitting_advancement=value['summary']['fitting_advancement'],new_training=False)))
