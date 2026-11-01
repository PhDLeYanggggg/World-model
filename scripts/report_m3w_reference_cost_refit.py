"""Reconstruct the reference-cost study from public scalar receipts only."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_reference_cost_refit as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def training_statistics(fits):
    reg=json.loads(run.REG.read_text());cfg=json.loads(run.CONFIG.read_text())
    expected={(c,s,a) for c in reg['contexts'] for s in cfg['seeds'] for a in run.api.ARMS}
    if len(fits)!=36 or {(f['context'],f['seed'],f['arm']) for f in fits}!=expected:
        raise ValueError('All36 distinct cost fits are required')
    for f in fits:
        budget=max(cfg['training']['minimum_updates'],cfg['training']['largest_source_cycles']*
            math.ceil(max(f['source_query_counts'])/(cfg['training']['query_batch_size']//4)))
        active=30857 if f['arm']=='locked_references' else 31119
        if (f['step']!=budget or f['budget']!=budget or f['parameters']!=31119
                or f['trainable_parameters']!=active or f['minimum_query_visits']<4
                or f['never_drawn_queries'] or f['unknown_rows_with_supervised_labels']):
            raise ValueError('Complete coverage,budget,capacity and honest unknown labels required')
    for c,s,_ in expected:
        a,b=[f for f in fits if (f['context'],f['seed'])==(c,s)]
        for k in ('input_hashes','draw_hash','source_query_counts','source_completed_cycles','warm_checkpoint','budget'):
            if a[k]!=b[k]:raise ValueError('Matched warm start,features,draws and budget required')
    arms={}
    for arm in run.api.ARMS:
        cells=[f for f in fits if f['arm']==arm]
        keys=cells[0]['first_trace']['monitor']
        arms[arm]=dict(fits=len(cells),updates=sum(f['step'] for f in cells),
            parameters=cells[0]['parameters'],trainable_parameters=cells[0]['trainable_parameters'],
            minimum_query_visits=min(f['minimum_query_visits'] for f in cells),
            never_drawn_queries=sum(f['never_drawn_queries'] for f in cells),
            unknown_rows_drawn=sum(f['unknown_rows_drawn'] for f in cells),
            first_monitor_mean={k:sum(f['first_trace']['monitor'][k] for f in cells)/len(cells) for k in keys},
            final_monitor_mean={k:sum(f['final_trace']['monitor'][k] for f in cells)/len(cells) for k in keys})
    return dict(complete_fits=36,primary_optimizer_updates=sum(f['step'] for f in fits),
        budgets=sorted(set(f['budget'] for f in fits)),arms=arms,
        active_capacity_matched=False,additional_active_reference_parameters=262,
        warm_start_not_from_scratch=True,unknown_rows_with_supervised_labels=0)


def supplementary(rows):
    tails={};pair={};support={};reasons={};nonoptimal={}
    total_pairs=sum(r['result']['pair_proxy']['pair_requested_step_views'] for r in rows)
    for arm in run.evaluation.NAMES:
        values=[v[arm] for r in rows for v in r['result']['recordings'].values()]
        known=[100*(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/
            v['full_known_easy_reference_mass'] for v in values if v['full_known_easy_reference_mass']>0]
        upper=[100*v['easy_degradation_upper'] for v in values if v['easy_degradation_upper'] is not None]
        tails[arm]=dict(required_record_views=len(values),known_defined=len(known),
            worst_known_easy_degradation_percent=max(known) if known else None,
            known_above2_percent=sum(v>2+1e-10 for v in known),
            completion_upper_defined=len(upper),worst_completion_upper_percent=max(upper) if upper else None)
        weighted=[]
        for row in rows:
            p=row['result']['pair_proxy'];n=p['pair_requested_step_views']
            if n:
                v=p['mean_positive_distance_contraction_over_frozen_scale'][arm]
                if v is None:raise ValueError('Observed pair support requires a value')
                weighted.append(n*v)
        p95=[v[arm]['p95'] for r in rows for v in r['result']['record_tails'].values() if v[arm]['p95'] is not None]
        pair[arm]=dict(pair_contraction_proxy=sum(weighted)/total_pairs if total_pairs else None,
            maximum_recording_p95=max(p95) if p95 else None)
    for arm in run.api.ARMS:
        counts={}
        for row in rows:
            for name,metrics in row['result']['pointwise_diagnostic'][arm]['supports'].items():
                dest=counts.setdefault(name,{k:0 for k in ('allowed_action_occurrences','supported_record_views','positive_record_views')})
                for k in dest:dest[k]+=metrics[k]
        support[arm]=counts
    for row in rows:
        for arm,n in row['result']['allocation']['nonoptimal'].items():nonoptimal[arm]=nonoptimal.get(arm,0)+n
        for arm,values in row['result']['allocation']['reason_counts'].items():
            dest=reasons.setdefault(arm,{})
            for reason,n in values.items():dest[reason]=dest.get(reason,0)+n
    return dict(easy_tails=tails,pair_and_tail=pair,pointwise_label_aware_diagnostic=support,
        query_views=sum(r['result']['allocation']['queries'] for r in rows),
        solver_calls=sum(r['result']['allocation']['solver_calls'] for r in rows),
        repeated_pair_requested_step_views=total_pairs,
        solver_reason_counts=reasons,nonoptimal_solver_queries=nonoptimal,
        oracle_diagnostic_not_model_gain=True,proxy_not_physical_safety=True)


def locality_diagnostics(rows,summary):
    output={}
    for source in sorted({r['source'] for r in rows}):
        group=[r for r in rows if r['source']==source];values={}
        for prefix,arm in zip(('locked','learned'),run.api.ARMS):
            records=[v[prefix+'_joint'] for row in group for v in row['result']['recordings'].values()]
            values[prefix]=dict(required_record_views=len(records),
                supported_record_views=sum(v['budget_status']['supported'] for v in records),
                undefined_easy_views=sum(v['easy_selected_risk_upper'] is None for v in records),
                pointwise_witness_views=sum(row['result']['pointwise_diagnostic'][arm]['supports']['reference_pointwise']['supported_record_views'] for row in group))
        output[source]=dict(joint_known_pp=summary['effects']['joint']['known_pp'].get('by_locality',{}).get(source),
            joint_completion_lower_pp=summary['effects']['joint']['lower_pp'].get('by_locality',{}).get(source),
            reference_MSE_change=summary['reference_MSE_change'].get('by_locality',{}).get(source),policies=values,
            descriptive_SOURCE_only_not_causal_attribution=True)
    return output


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text());cfg=json.loads(run.CONFIG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Registered scientific definition changed')
    from scripts.run_m3w_reference_cost_refit_readout import corrected_evaluation
    erratum=json.loads((p/'execution_erratum.json').read_text())
    if (erratum['binding']!=corrected_evaluation()[1] or erratum['new_policy_groups_created_before_fix']!=0
            or not erratum['registered_before_own_policy_outcomes']
            or erratum['training_freeze_sha256']!=run.sha(p/'training_freeze.json')):
        raise ValueError('Only the pre-outcome scope-name erratum is allowed')
    freeze=json.loads((p/'training_freeze.json').read_text());fits=[]
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Original fit receipt changed')
        f=json.loads((ROOT/ref['path']).read_text())
        if f['registration_sha256']!=run.sha(run.REG):raise ValueError('Fit registration differs')
        fits.append(f)
    stats=training_statistics(fits);refits={}
    coherence=json.loads((p/'coherence_verification.json').read_text())
    expected={f'{f["context"]}_seed{f["seed"]}_{f["arm"]}':f for f in fits}
    if (set(coherence['cells'])!=set(expected) or not coherence['prediction_bytes_unchanged']
            or coherence['registration_sha256']!=run.sha(run.REG)
            or coherence['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
            or coherence['verification_sha256']!=run.sha(p/'verification.json')
            or coherence['executable_sha256']!=run.sha(ROOT/'scripts/verify_m3w_reference_refit_coherence.py')
            or coherence['future_targets_read_by_diagnostic'] or coherence['thresholds_or_choices_changed']):
        raise ValueError('Nonintervening coherence diagnostic must retain original predictions and verification')
    for key,f in expected.items():
        if coherence['cells'][key]['checkpoint_sha256']!=f['checkpoint']['sha256']:
            raise ValueError('Coherence diagnostic must use the same36frozen weights')
    first=sorted(reg['contexts'])[0]
    for arm in run.api.ARMS:
        r=json.loads((p/f'first_full_refit_{arm}.json').read_text())
        fit=next(f for f in fits if (f['context'],f['seed'],f['arm'])==(first,17,arm))
        if not r['scientific_state_exact'] or r['updates']!=fit['budget'] or r['checkpoint_sha256']!=fit['checkpoint']['sha256']:
            raise ValueError('Both complete full-budget refits must be exact')
        refits[arm]=r
    stats['exact_full_refit_updates']=sum(v['updates'] for v in refits.values())
    if (freeze['registration_sha256']!=run.sha(run.REG)
            or freeze['primary_optimizer_updates']!=stats['primary_optimizer_updates']
            or freeze['exact_full_refit_updates']!=stats['exact_full_refit_updates']):
        raise ValueError('Original full training freeze differs')
    bundle=verify(ROOT,p);complete=json.loads((p/'complete.json').read_text());replay=json.loads((p/'verification.json').read_text())
    if (complete['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
            or complete['summary_sha256']!=run.sha(p/'summary.json')
            or replay['summary_sha256']!=run.sha(p/'summary.json')
            or replay['complete_sha256']!=run.sha(p/'complete.json')
            or not replay['all72_new_cost_inference_frozen_actions_and_scores_exact']
            or not replay['original_parent_controls_exact'] or replay['optimization_replayed']):
        raise ValueError('Complete original readout and separate inference/score verification required')
    old=aggregate_groups(run.parent.NAME);groups=aggregate_groups(run.NAME)
    rows=[groups[Path(r['path']).name] for r in complete['groups']];recordings=set()
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json';parent=old[key]
        if (row['ids_sha256']!=parent['ids_sha256']
                or row['parent_prediction_sha256']!=parent['cost_prediction_sha256']
                or row['action_sha256']['parent_joint']!=parent['action_sha256']['zero_joint']
                or row['result']['policies']['parent_joint']!=parent['result']['policies']['zero_joint']
                or row['registration_sha256']!=run.sha(run.REG)
                or row['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
                or not row['original_parent_exact'] or not row['matched_training']):
            raise ValueError('Original rows,reference policy or frozen models changed')
        recordings.update((row['source'],r) for r in row['result']['recordings'])
    summary=json.loads((p/'summary.json').read_text());rebuilt=run.evaluation.summarize(rows,reg['contexts'],cfg)
    rebuilt.update(unique_target_rows=149510,unique_recordings=len(recordings))
    if summary!=rebuilt or len(recordings)!=74:raise ValueError('Full canonical summary differs')
    resource=json.loads((p/'verification_resource.json').read_text())
    if resource['verification_sha256']!=run.sha(p/'verification.json') or resource['optimization_replayed']:
        raise ValueError('Verification resource receipt must refer to the same frozen-action replay')
    return dict(summary=summary,complete=complete,freeze=freeze,replay=replay,bundle=bundle,verification_resource=resource,
        training=stats,refits=refits,auxiliary=supplementary(rows),coherence=coherence,
        locality_diagnostics=locality_diagnostics(rows,summary),
        pilot=json.loads((p/'pilot.json').read_text()))


def cell(v):return 'not_estimable' if v is None else f'{v:+.6f}'


def interval(v):
    return 'not_estimable' if v['CI95'] is None else f'{v["point"]:+.6f} [{v["CI95"][0]:+.6f},{v["CI95"][1]:+.6f}]'


def render(v):
    p=run.PUBLIC;s=v['summary'];stats=v['training'];aux=v['auxiliary']
    e=s['effects']['joint'];penalty=None if any(e[k]['point'] is None for k in ('known_pp','lower_pp')) else e['known_pp']['point']-e['lower_pp']['point']
    run.write(p/'training_statistics.json',stats);run.write(p/'supplementary_statistics.json',aux)
    run.write(p/'locality_diagnostics.json',v['locality_diagnostics'])
    lines=['# Coverage-Complete Reference Cost Refits','','## Evidence Role','',
        'fresh_run:36 actual Torch warm-started cost fits,two full-budget exact refits,complete72-group policy evaluation and separate inference/score verification.',
        'cached_verified:18 old cost weights,seven forecast candidates,floor,462causal fields,fitting normalizers and original parent full-joint control.',
        'not_run:new forecaster/JEPA/Transformer/residual training,independent selection/calibration/confirmation,whole-pipeline OOF or deployment.',
        'All present outcomes are exposed SOURCE fitting/resubstitution. They are not independent test results.','',
        '## Prespecified Primary','',
        'Learned-reference minus locked-reference FULL joint completion-lower ADE utility is the primary.',
        'Both use the same warm start,query draws,budget and31,119total parameters. The learned arm has262additional active parameters;',
        'this is not a fully active-capacity-matched ablation. The locked continuation combines changed coverage and longer budgets relative to the old random2,000-update model.','',
        '|Policy contrast|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|Completion-upper pp [conditional95%CI]|',
        '|---|---:|---:|---:|']
    for mode,e in s['effects'].items():lines.append(f'|learned minus locked:{mode}|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    e=s['parent_effect'];lines.append(f'|learned joint minus original parent|{interval(e["known_pp"])}|{interval(e["lower_pp"])}|{interval(e["upper_pp"])}|')
    lines+=['',f'Paired known-minus-completion-lower penalty:{cell(penalty)}pp. This is missing-outcome uncertainty,not observed harm.',
        'Count-matched controls evaluate utility only. Empty selected easy support stays not_estimable,not safe0.','',
        '|Policy|Known ADE gain over floor (%)|Known FDE gain (%)|Own completion-lower pp|Supported original record-risk views|Known easy-risk violations|Undefined selected easy views|Selected / unknown views|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        r=s['safety'][a];lines.append(f'|{a}|{interval(s["accuracy"][a])}|{interval(s["FDE"][a])}|{interval(s["floor_lower"][a])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{r["selected"]} / {r["selected_unknown"]}|')
    lines+=['','## Cost Prediction Error','','Normalized SOURCE fitting MSE,not independent calibration. Indices0..4 are benefit,harm,reference,easy reference,easy harm.','',
        '|Moment|Locked [conditional95%CI]|Learned [conditional95%CI]|','|---|---:|---:|']
    for k in range(5):lines.append(f'|{k}|{interval(s["calibration_MSE"]["locked_references"][str(k)])}|{interval(s["calibration_MSE"]["learned_references"][str(k)])}|')
    lines += ['',f'Mean reference MSE change:{interval(s["reference_MSE_change"])}.','',
        '## Prediction Coherence Diagnostic','',
        'Added after registration,not a new primary or gate. No labels are read by this diagnostic and not one prediction byte or action changes.',
        'Counts repeat source/context/cost-seed views;they are not independent samples. Excess tolerance is1e-12in frozen causal scale units.','',
        '|Arm|Legal action views|Benefit above reference|Negative implied candidate ADE|Easy reference above reference|Easy harm above harm|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        cells=[x['diagnostic'] for x in v['coherence']['cells'].values() if x['arm']==arm]
        numbers=[sum(x['legal_action_views'] for x in cells)]+[sum(x['violations'][k]['above1e_12_normalized_action_views'] for x in cells)
            for k in ('benefit_exceeds_reference','negative_implied_candidate_ADE','easy_reference_exceeds_reference','easy_harm_exceeds_harm')]
        lines.append('|'+arm+'|'+'|'.join(map(str,numbers))+'|')
    lines+=['','Per-model exact/tolerance violations,zero references and candidate-reference spread remain in coherence_verification.json.',
        'A coherent prediction is not proof of calibrated risk or useful decisions;incoherent moments locate a concrete remaining model limitation.','',
        '## Source-Level Diagnostic','','These repeat contexts and cost seeds,not independent recording samples. Cost-MSE changes are descriptive,not causal attribution.','',
        '|Exposed locality|Learned-minus-locked lower pp|Reference MSE change|Locked record-risk support|Learned record-risk support|Locked pointwise witnesses|Learned pointwise witnesses|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for source,d in v['locality_diagnostics'].items():
        a,b=d['policies']['locked'],d['policies']['learned']
        lines.append(f'|{source}|{cell(d["joint_completion_lower_pp"])}|{cell(d["reference_MSE_change"])}|{a["supported_record_views"]}/{a["required_record_views"]}|{b["supported_record_views"]}/{b["required_record_views"]}|{a["pointwise_witness_views"]}|{b["pointwise_witness_views"]}|')
    lines += ['',
        '## Complete Training And Coverage','',
        f'{stats["complete_fits"]}fits;{stats["primary_optimizer_updates"]}primary optimizer updates;{stats["exact_full_refit_updates"]}full-refit updates.',
        f'Registered per-fit budgets:{stats["budgets"]}. Every query enters at least4optimizer-step sampling batches;zero queries remain undrawn.',
        'Unknown outcomes never become supervised zeros. Unknown rows may enter causal query context but receive no cost-label gradient.','',
        '|Arm|Updates|Active parameters|First fixed-monitor loss|Final fixed-monitor loss|First reference MSE|Final reference MSE|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,t in stats['arms'].items():
        x,y=t['first_monitor_mean'],t['final_monitor_mean']
        lines.append(f'|{a}|{t["updates"]}|{t["trainable_parameters"]}|{x["total"]:.6f}|{y["total"]:.6f}|{x["reference_MSE"]:.6f}|{y["reference_MSE"]:.6f}|')
    lines+=['','## Original Easy Tails','','|Policy|Known easy views|Worst known degradation (%)|Known views above2%|Worst completion-upper bound (%)|',
        '|---|---:|---:|---:|---:|']
    for a,t in aux['easy_tails'].items():lines.append(f'|{a}|{t["known_defined"]}/{t["required_record_views"]}|{cell(t["worst_known_easy_degradation_percent"])}|{t["known_above2_percent"]}|{cell(t["worst_completion_upper_percent"])}|')
    lines+=['','Bounds on missing outcomes are not observed errors. Pooled hard/easy results cannot replace every original recording requirement.','',
        '|Known-label subset|Original parent|Locked joint|Learned joint|','|---|---:|---:|---:|']
    for subset in ('hard','easy'):
        x=s['subsets'][subset]['policies'];lines.append('|'+subset+'|'+'|'.join(cell(x[a]['ADE_gain_percent']) for a in ('parent_joint','locked_joint','learned_joint'))+'|')
    lines+=['','## Resource And Recovery Evidence','',
        f'Pilot PID{v["pilot"]["pid"]}:{v["pilot"]["seconds"]:.6f}s,peakRSS{v["pilot"]["peak_RSS_bytes"]}B;400actual updates,zero initialization and interruption recovery exact.',
        f'Training PID{v["freeze"]["pid"]}:{v["freeze"]["seconds"]:.6f}s,peakRSS{v["freeze"]["peak_RSS_bytes"]}B.',
        f'Policy readout PID{v["complete"]["pid"]}:{v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B.',
        f'Separate verification PID{v["verification_resource"]["pid"]}:{v["verification_resource"]["seconds"]:.6f}s,peakRSS{v["verification_resource"]["peak_RSS_bytes"]}B.',
        f'Complete readout:{aux["query_views"]}repeated query views,{aux["solver_calls"]}solver calls. Phase timers exclude admission.',
        'Both first-context full-budget refits are exact. Separate72-group verification re-infers costs and scores frozen actions.',
        'It does not rerun wall-clock-limited MILP optimization. Actual allocation checkpoint recovery has separate focused tests.',
        'The original evaluation entry failed before runtime_ready with a local runtime-name collision. A committed execution erratum renames only that receipt binding and its two uses;all original scientific files and models remain unchanged.',
        'Solver reason/nonoptimal counts are retained in supplementary_statistics.json;nonoptimal calls keep their feasible independent parent.',
        'Native arm64CPU4/interop1/workers0,checkpoint100/heartbeat50,no resource probing. No reduced roster or speed-based downgrade.','',
        '## Diagnostic Support And Interaction Proxy','',
        'Pointwise label-aware support counts are saved in supplementary_statistics.json. They use labels for diagnosis,not inference or real utility.',
        'Pair contraction is a query-target diagnostic,not collision safety. Recording-p95 maximum is not a pooled percentile.','',
        '|Policy|Pair contraction / frozen scale|Maximum recording p95 ADE|','|---|---:|---:|']
    for a in ('parent_joint','locked_joint','learned_joint','floor'):
        t=aux['pair_and_tail'][a];lines.append(f'|{a}|{cell(t["pair_contraction_proxy"])}|{cell(t["maximum_recording_p95"])}|')
    lines+=['','## Fitting Gates','']+[f'- {k}: {"pass" if x else "fail"}' for k,x in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement"]}. No deployment or independent confirmation.','',
        '![Reference cost study](figures/reference_cost_readout.svg)','',
        '## Limits','',
        'All149,510unique targets,74recordings,72source/context/critic-seed views and12exposed localities remain.',
        '3,000paired locality bootstrap draws average repeated contexts and cost seeds before resampling. Intervals are nominal,conditional fitting evidence.',
        'Cost seeds17/29/43share forecaster43,not three whole forecast pipelines. Overlapping windows are not independent samples.',
        'Full-roster FDE remains not_estimable without requested endpoints;no earlier endpoint substitution. Missing interior labels remain unresolved.',
        'Reference updates share a row scale but preserve old per-action differences. Exact-zero references cannot become positive multiplicatively.',
        'Inherited predicted benefit<=reference is not structurally guaranteed. Lower reference MSE does not establish decision-ratio calibration.',
        'EuropeanSquares oblique-camera detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Stage5C and SMC remain disabled. Not yet a CCF-A submission candidate.','']
    (p/'readout.md').write_text('\n'.join(lines))
    (p/'gates.md').write_text('# Reference Cost Fitting Gates\n\n'+'\n'.join(f'- {k}: {x}' for k,x in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement"]}. Independent confirmation:false. Deployment:false. Submission ready:false. Stage5C:false. SMC:false.\n')
    with (p/'locality_effects.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['locality','joint_known_pp','joint_lower_pp','joint_upper_pp'])
        for source in v['locality_diagnostics']:
            w.writerow([source]+[s['effects']['joint'][b].get('by_locality',{}).get(source,'not_estimable') for b in ('known_pp','lower_pp','upper_pp')])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),complete_sha256=run.sha(p/'complete.json'),
        training_freeze_sha256=run.sha(p/'training_freeze.json'),verification_sha256=run.sha(p/'verification.json'),
        coherence_sha256=run.sha(p/'coherence_verification.json'),verification_resource_sha256=run.sha(p/'verification_resource.json'),
        archive_sha256=v['bundle']['archive_sha256'],fresh_cost_training=True,new_trajectory_training=False,
        fitting_advancement=s['fitting_advancement'],independent_confirmation=False,deployment_changed=False,submission_ready=False))


def interpretation(v):
    s=v['summary'];p=run.PUBLIC;r=s['safety']['learned_joint'];e=s['effects']['joint'];t=v['auxiliary']['easy_tails']['learned_joint']
    co={}
    for arm in run.api.ARMS:
        ds=[x['diagnostic'] for x in v['coherence']['cells'].values() if x['arm']==arm]
        co[arm]=dict(legal=sum(d['legal_action_views'] for d in ds),
            negative_ADE=sum(d['violations']['negative_implied_candidate_ADE']['above1e_12_normalized_action_views'] for d in ds),
            B_above_R=sum(d['violations']['benefit_exceeds_reference']['above1e_12_normalized_action_views'] for d in ds),
            multiaction_rows=sum(d['reference_spread']['reference']['rows_with_two_legal_actions'] for d in ds),
            inconsistent_reference_rows=sum(d['reference_spread']['reference']['rows_with_spread_above1e_12'] for d in ds),
            zero_R=sum(d['exact_zero_reference_action_views'] for d in ds),
            zero_ER=sum(d['exact_zero_easy_reference_action_views'] for d in ds))
    locked,learned=co['locked_references'],co['learned_references']
    defined=[(source,d) for source,d in v['locality_diagnostics'].items()
        if d['joint_known_pp'] is not None and d['joint_completion_lower_pp'] is not None]
    worst=max(defined,key=lambda item:item[1]['joint_known_pp']-item[1]['joint_completion_lower_pp']) if defined else None
    lines=['# Cost Repair: Failure Taxonomy And Interpretation','',
        'This follows the complete registered readout. It does not select a new primary,seed,threshold or recording subset.','',
        '|Question|Measured evidence|Conclusion boundary|','|---|---|---|',
        f'|Sampling coverage repaired?|All36 fits have minimum4query visits and zero undrawn queries|Repairs sampling coverage;wholly unknown labels still provide no supervised gradient. Not a generalization claim.|',
        f'|Reference learning helps the registered decision target?|Known:{interval(e["known_pp"])};completion-lower:{interval(e["lower_pp"])}|Use the lower-CI gate,not the favorable point estimate alone.|',
        f'|Reference prediction improves?|Normalized reference MSE change:{interval(s["reference_MSE_change"])}|Fitting error is not independent calibration;ratio or sign errors may remain.|',
        f'|Reference-target consistency?|Locked:{locked["inconsistent_reference_rows"]}/{locked["multiaction_rows"]};learned:{learned["inconsistent_reference_rows"]}/{learned["multiaction_rows"]} multi-action row views have differing predicted R|The actual reference target is action-invariant;row-shared scaling has not repaired that structural inconsistency.|',
        f'|Implied candidate error coherence?|Negative expected ADE from moments:locked{locked["negative_ADE"]}/{locked["legal"]};learned{learned["negative_ADE"]}/{learned["legal"]}|These are cost-model contradictions,not negative measured trajectory errors. They increase despite lower reference MSE;selected-policy causation is not established.|',
        f'|Benefit-reference ordering?|B>R:locked{locked["B_above_R"]};learned{learned["B_above_R"]} legal action views|The inherited coupled head does not enforce B<=R. Both positive and harm expectations still need representation;netting them away would hide risk.|',
        f'|Original full-policy risk?|Supported {r["supported_record_views"]}/{r["required_record_views"]};undefined easy:{r["undefined_easy_record_views"]};known easy-risk violations:{r["known_easy_risk_violations"]}|No full-policy safety pass from pooled gains or count-matched controls.|',
        f'|Easy accuracy tails?|Worst known degradation:{cell(t["worst_known_easy_degradation_percent"])}%;above2%:{t["known_above2_percent"]}|Do not hide individual recording harm.|',
        '|Capacity attribution|262additional active parameters in the learned arm|Not fully active-capacity matched;the changed branch is not proof of a general architecture contribution.|',
        '|Baseline continuation attribution|Both arms change coverage and continue the old fit with reset optimizers|Cannot isolate coverage alone versus extra updates relative to the historical parent.|',
        f'|Exact-zero reference explanation?|Locked R/ER zeros:{locked["zero_R"]}/{locked["zero_ER"]};learned:{learned["zero_R"]}/{learned["zero_ER"]}|Multiplicative zero recovery is a theoretical limitation,but not an observed cause in these legal action views.|',
        '|Missing labels|Original conservative bounds and full roster retained|Unknown disagreement is uncertainty,not observed harm or a fabricated zero.|',
        '|Generalization|Exposed SOURCE fitting,one forecast producer|No independent confirmation,cross-dataset success or submission readiness.|','',
        f'Fitting advancement:{s["fitting_advancement"]}. Deployment stays unchanged.','']
    (p/'failure_analysis.md').write_text('\n'.join(lines))
    details=['# Measured Diagnosis And Next Controlled Test','',
        'This is post-readout SOURCE diagnosis,not a new primary or a deployment policy. All original outcomes remain frozen.','',
        '## What Was Repaired','',
        'All original queries are now sampled at least four times in each of36 actual optimizer fits. No SOURCE query is undrawn.',
        'All12exposed localities have lower reference fitting MSE in the learned arm. The known-label joint effect is positive with its conditional interval.',
        'The previous sampler defect and frozen reference learning are therefore addressed operationally;this does not establish independent efficacy.','',
        '## What Still Fails','',
        f'Paired full completion-lower effect:{interval(e["lower_pp"])}pp;even count-matched lower:{interval(s["effects"]["matched"]["lower_pp"])}pp.',
        f'Known-minus-lower uncertainty penalty:{e["known_pp"]["point"]-e["lower_pp"]["point"]:.6f}pp. This is uncertainty about unobserved outcomes,not measured damage.',
        (f'Largest paired uncertainty penalty:{worst[0]},known{cell(worst[1]["joint_known_pp"])}pp versus lower{cell(worst[1]["joint_completion_lower_pp"])}pp. This is not measured harm.' if worst else 'Source-specific uncertainty is not_estimable.'),
        f'Actual known all/easy positive-risk violations:{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]} repeated record views. These are distinct from uncertainty.',
        f'Worst known easy degradation:{t["worst_known_easy_degradation_percent"]:.6f}%;{t["known_above2_percent"]}views exceed2%. The full policy cannot be promoted.',
        'Normalized benefit and harm fitting MSE do not improve with the reference-only auxiliary objective. All five moment errors are retained in the main table.',
        f'Candidate-reference inconsistency persists in all{learned["multiaction_rows"]}legal multi-action row views. Negative implied candidate ADE rises from{locked["negative_ADE"]}to{learned["negative_ADE"]}.',
        'This locates a real model-class limitation,but does not prove it accounts for the entire policy effect. Coherence diagnostics do not use targets or change actions.',
        'The exact-zero R/ER explanation has no observed support here. Remaining pointwise witness failures are not a proof that query pooling is infeasible.',
        'Nonoptimal solver queries are few and retain their prespecified feasible parent;counts are public. A different optimization budget was not selected after results.','',
        '## Next Minimal Discriminating Experiment','',
        'Test an explicitly causal coherent-cost projection before another larger training run:share R and ER across an agent\'s candidates,',
        'enforce B<=R and nonnegative implied expected error,retain separate positive benefit/harm and the original envelope bounds.',
        'Use the same frozen forecasts,features,rows and2%constraints. Compare zero-step projection and unprojected controls on all72SOURCE groups.',
        'Register the new definition before its own policy outcomes;do not reuse a favorable seed or subset as the new primary.',
        'A projection changes initialization and decisions:it must not be described as exact old-model reproduction or new neural training.',
        'If helpful,train the coherent decoder with a matched initialization/coverage budget and a reference-learning control;measure all five costs and actual recording risk.',
        'If insufficient,inspect selected-query/record risk objectives and outcome-label coverage rather than merely extending the same reference-MSE fit.',
        'The current SOURCE evidence cannot distinguish all uncertainty/calibration mechanisms. No guaranteed success is predicted.',
        'No independent labels are opened to rescue this result. Role-compliant producer training,independent calibration/confirmation and contribution baselines remain missing.',
        'Stage5C and SMC remain disabled. The existing deployment policy is unchanged.','']
    (p/'diagnosis_and_next_control.md').write_text('\n'.join(details))
    text=['# Publication Gap And Next Experiment','',
        'Completed:actual continuous-cost training,coverage-controlled paired continuation,complete policy/readout,',
        'two exact full-budget refits,conditional paired bootstrap and public scalar reconstruction.','',
        'Still required:a useful method effect under every original recording-risk requirement,independent model selection/calibration/confirmation,',
        'role-compliant full-pipeline training,matched strong selective/decision-aware baselines,scene/interaction ablations,adequate label support,',
        'independent contribution evidence,a complete anonymous paper/supplement and reproduction.','',
        'If the registered fitting gate fails,diagnose cost-ratio/sign errors and original record support from the frozen outputs before another training choice.',
        'Do not repeatedly open independent labels to rescue a source fit. If it passes,freeze the complete pipeline before authorized independent-role evaluation.',
        'The original2%risk requirements and undefined-support rule remain. No Stage5C or SMC execution.',
        'European local-image/raw-frame detector-silver evidence is not metric,seconds,true3D or foundation. Not yet a CCF-A submission candidate.','']
    (p/'publication_gap.md').write_text('\n'.join(text))
    card=['# Model And Data Card','',
        'Purpose:predict intervention benefit,harm,reference and easy-risk moments for a fixed seven-forecast bank.',
        'Model:31,119parameter causal query cost head,462input fields,warm-started from18verified released-benefit critics.',
        'Reference branch:262parameters,locked or learnable,causal row-shared multiplicative scaling. Not a trajectory/world-dynamics generator.',
        'Data:exposed EuropeanSquares SOURCE fittings,149,510unique targets,74recordings,12localities,obs8/pred12/rawstride12.',
        'Labels:detector-silver and partial outcomes,not human gold. Unknown-cost supervision is masked;missing requested endpoints remain missing.',
        'Inputs:no future endpoint,central velocity,test goals or independent-role outcome inputs. Future labels are used only for training loss/readout.',
        'Deployment:none changed. No physical-safety guarantee,independent efficacy,true3D,foundation,metric or seconds-level claim.',
        'Cost seeds17/29/43share forecaster43. Both fitted arms use the same complete cyclic query draws,budget and frozen normalizers.',
        'Limitations:SOURCE reuse,missing interior labels,active-capacity mismatch,inherited zero references,per-action reference inconsistency and ratio calibration.',
        'Private weights and raw/cache data are not in Git. Public hashes,fit receipts,scalar bundles and reproduction commands are provided.','']
    (p/'model_data_card.md').write_text('\n'.join(card))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors=('#326a95','#9a6a16','#b85042')
    for i,(name,color) in enumerate(zip(('parent_joint','locked_joint','learned_joint'),colors)):
        z=s['accuracy'][name];ci=z['CI95'];axes[0].bar(i,z['point'],width=.6,color=color)
        if ci is not None:axes[0].errorbar(i,z['point'],yerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='none',ecolor='#222222',capsize=5)
    axes[0].set_xticks(range(3),['Original parent','Locked refits','Learned references']);axes[0].set_ylabel('Known ADE gain over floor (%)')
    axes[0].set_title('Fixed forecasts; complete query cost training',fontsize=11)
    for i,b in enumerate(('known_pp','lower_pp')):
        z=s['effects']['joint'][b];ci=z['CI95']
        if ci is not None:axes[1].errorbar(z['point'],i,xerr=[[z['point']-ci[0]],[ci[1]-z['point']]],fmt='o',color=colors[i],capsize=5)
    axes[1].axvline(0,color='#333333',lw=.8);axes[1].set_yticks([0,1],['Known labels','Completion lower'])
    axes[1].set_xlabel('Learned minus locked joint utility (pp)');axes[1].set_title('Paired conditional 95% intervals',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE fitting only: 12 localities; no deployment',fontsize=12)
    dest=run.PUBLIC/'figures';dest.mkdir(exist_ok=True);svg=dest/'reference_cost_readout.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    run.PRIVATE.mkdir(parents=True,exist_ok=True)
    fig.savefig(run.PRIVATE/'reference_cost_readout.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    value=build();render(value);interpretation(value);figure(value)
    print(json.dumps(dict(groups=value['summary']['groups'],fits=value['training']['complete_fits'],
        primary=value['summary']['effects']['joint']['lower_pp'],advancement=value['summary']['fitting_advancement'])))
