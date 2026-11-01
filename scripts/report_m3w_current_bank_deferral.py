"""Reconstruct current-bank published-loss evidence from public scalar artifacts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_current_bank_deferral as run
from src.evaluation.m3w_metric_bundle import verify

cell,interval=run.parent_report.cell,run.parent_report.interval


def training():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text());freeze=json.loads((p/'training_freeze.json').read_text())
    expected={(c,s,a) for c in reg['contexts'] for s in reg['seeds'] for a in run.api.ARMS};cells={}
    parents={}
    for ref in json.loads((run.parent.PUBLIC/'training_freeze.json').read_text())['fits']:
        r=json.loads((ROOT/ref['path']).read_text())
        if r['arm']=='own_choice':parents[r['context'],r['seed']]=r
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Public fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text());key=r['context'],r['seed'],r['arm']
        if (key in cells or r['registration_sha256']!=run.sha(run.REG) or r['step']!=r['budget']
                or r['minimum_query_visits']<4 or r['parameters']!=30726
                or r['unknown_rows_with_supervised_labels'] or not r['prior_query_draws_exact']):
            raise ValueError('Complete matched real training required')
        old=parents[r['context'],r['seed']]
        for k in ('draw_hash','step','budget','unknown_rows_drawn','warm_checkpoint'):
            if r[k]!=old[k]:raise ValueError('Original matched draw/budget/warm source changed')
        if any(r['input_hashes'][k]!=v for k,v in old['input_hashes'].items()):raise ValueError('Original causal inputs or targets changed')
        cells[key]=r
    if set(cells)!=expected or len(cells)!=36 or freeze['registration_sha256']!=run.sha(run.REG):raise ValueError('All36registered fits required')
    for c in reg['contexts']:
        for s in reg['seeds']:
            a,b=(cells[c,s,x] for x in run.api.ARMS)
            for k in ('input_hashes','draw_hash','budget','parameters','source_query_counts','source_completed_cycles','warm_checkpoint'):
                if a[k]!=b[k]:raise ValueError('Paired loss-control inputs/sampling changed')
    updates=sum(r['step'] for r in cells.values());refits=0
    for arm in run.api.ARMS:
        r=json.loads((p/f'first_full_refit_{arm}.json').read_text())
        if not r['scientific_state_exact'] or not r['wall_time_excluded']:raise ValueError('Full-budget scientific refits required')
        refits+=r['updates']
    if updates!=208320 or freeze['primary_optimizer_updates']!=updates or refits!=14336 or freeze['exact_full_refit_updates']!=refits:
        raise ValueError('Original complete update budgets required')
    totals={a:dict(fits=18,updates=sum(r['step'] for r in cells.values() if r['arm']==a),
        unknown_rows_drawn=sum(r['unknown_rows_drawn'] for r in cells.values() if r['arm']==a),
        first_monitor_loss=sum(r['first_trace']['monitor_loss'] for r in cells.values() if r['arm']==a)/18,
        final_monitor_loss=sum(r['final_trace']['monitor_loss'] for r in cells.values() if r['arm']==a)/18) for a in run.api.ARMS}
    return dict(freeze=freeze,totals=totals)


def auxiliary(rows):
    result={}
    for name in run.evaluation.NAMES:
        p=[r['result']['policies'][name] for r in rows];c=[v[name] for r in rows for v in r['result']['recordings'].values()]
        counts={str(k):sum(v['action_counts'][str(k)] for v in p) for k in range(-1,7)};total=sum(counts.values())
        easy=[100*(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/v['full_known_easy_reference_mass']
              for v in c if v['full_known_easy_reference_mass']>0]
        result[name]=dict(action_counts=counts,repeated_row_views=total,switch_rate=1-counts['-1']/total,
            neural_row_rate=counts['6']/total,group_mean_normalized_regret=float(run.api.np.mean([v['known_regret_over_cost_scale'] for v in p])),
            worst_known_easy_degradation_percent=max(easy) if easy else None,known_easy_views_above2=sum(v>2+1e-10 for v in easy),
            required_record_views=len(c),known_easy_defined=len(easy))
    return result


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text());cfg=json.loads(run.CONFIG.read_text())
    for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Registered science changed')
    for name,digest in reg['parent_files'].items():
        if run.sha(run.parent.PUBLIC/name)!=digest:raise ValueError('Frozen parent evidence changed')
    run.parent_report.build();trained=training();bundle=verify(ROOT,p)
    complete=json.loads((p/'complete.json').read_text());replay=json.loads((p/'verification.json').read_text())
    if (complete['training_freeze_sha256']!=run.sha(p/'training_freeze.json') or complete['summary_sha256']!=run.sha(p/'summary.json')
            or replay['summary_sha256']!=complete['summary_sha256'] or replay['complete_sha256']!=run.sha(p/'complete.json')
            or not replay['all72_new_score_inference_and_action_scores_exact'] or not replay['original_full_and_record_controls_exact']):
        raise ValueError('Complete readout and actual numerical replay required')
    groups=run.parent.aggregate_groups(run.NAME);old=run.parent.aggregate_groups(run.parent.NAME);records=set()
    rows=[groups[Path(ref['path']).name] for ref in complete['groups']]
    for r in rows:
        prior=old[f'{r["context"]}_{r["source"]}_seed{r["seed"]}_fit.json']
        if (r['registration_sha256']!=run.sha(run.REG) or r['training_freeze_sha256']!=run.sha(p/'training_freeze.json')
                or r['parent_complete_sha256']!=run.sha(run.parent.PUBLIC/'complete.json') or r['ids_sha256']!=prior['ids_sha256']):
            raise ValueError('Original group/input lineage changed')
        for mode in ('independent','joint'):
            a='own_'+mode;b='frozen_'+a
            if r['action_sha256'][b]!=prior['action_sha256'][a] or r['result']['policies'][b]!=prior['result']['policies'][a]:
                raise ValueError('Frozen full control differs')
            for rec,v in r['result']['recordings'].items():
                if v[b]!=prior['result']['recordings'][rec][a]:raise ValueError('Frozen recording control differs')
        records.update((r['source'],k) for k in r['result']['recordings'])
    summary=run.evaluation.summarize(rows,reg['contexts'],cfg);summary.update(unique_target_rows=149510,unique_recordings=len(records))
    if summary!=json.loads((p/'summary.json').read_text()) or len(records)!=74:raise ValueError('Original full72canonical summary required')
    return dict(summary=summary,training=trained,complete=complete,replay=replay,bundle=bundle,auxiliary=auxiliary(rows),pilot=json.loads((p/'pilot.json').read_text()))


def render(v):
    p=run.PUBLIC;s=v['summary'];a=v['auxiliary'];t=v['training'];run.write(p/'supplementary_statistics.json',a)
    lines=['# Current-Bank ADE Regression Versus Published Deferral','','## Material Passport','',
        'fresh_run:36real Torch fits,complete72-group policy readout and separate new-score/action-score verification.',
        'cached_verified:seven current forecasts,462causal fields,normalizers,warm encoder and original own-choice risk critic/controls.',
        'not_run:new trajectory/JEPA/Transformer/residual training,independent model selection/calibration/confirmation and deployment.',
        'All results are exposed SOURCE fitting/resubstitution. Three cost seeds share forecaster43.','',
        '## Controlled Comparison','',
        'Same30,726active parameters,initial encoder and zero outputs,query draws,full budgets and canonical continuous expert labels.',
        'Expected-ADE softplus MSE versus the reused Mao/Mohri/Zhong ICML2024Eq3logistic surrogate. No one-hot best-action labels.',
        'The architecture/floor pooling is an application adapter,not their original experimental reproduction or a bounded-loss theorem verification.',
        'Raw scores select independently. The guarded version filters by the frozen own critic\'s original two2%pointwise predicted constraints and positive gain.',
        'No new threshold,calibration,integer solver or logit-as-native-utility substitution. No future label/validity enters decisions.',
        'The previous31,119parameter critic is a different head family;the new two-loss control has exactly matched capacity internally.','',
        '|Loss|Fits|Updates|Initial monitor loss|Final monitor loss|Unknown contextual row draws|',
        '|---|---:|---:|---:|---:|---:|']
    for arm,x in t['totals'].items():lines.append(f'|{arm}|{x["fits"]}|{x["updates"]}|{cell(x["first_monitor_loss"])}|{cell(x["final_monitor_loss"])}|{x["unknown_rows_drawn"]}|')
    lines+=['','Loss magnitudes have different objectives and are not a cross-loss performance ranking. Full endpoint budgets are fixed,not validation-selected checkpoints.',
        'Every query receives at least four visits. Unknown labels stay NaN;unknown rows retain causal contextual roles but no supervised cost.',
        'All208,320primary updates plus14,336full-refit updates complete. Actual interrupted/direct and full-refit scientific states are exact,excluding wall time.','',
        '## Registered Contrasts','',
        'Primary:TDEF_guarded minus ADE_guarded FULL completion-lower utility. No raw/component/seed substitutes for it.',
        '|Contrast|Scope|Known pp [conditional95%CI]|Completion-lower pp [conditional95%CI]|Completion-upper pp [conditional95%CI]|',
        '|---|---|---:|---:|---:|']
    for name,effects in s['effects'].items():
        for scope,x in effects.items():lines.append(f'|{name}|{scope}|{interval(x["known_pp"])}|{interval(x["lower_pp"])}|{interval(x["upper_pp"])}|')
    lines+=['','Post-hoc common-count views are utility-only,using each policy\'s own score margins;empty required views remain not_estimable,not safe0.',
        'Paired lower uses unknown-row trajectory disagreement,not subtraction of two individual lower scores. Uncertainty is not observed hidden harm.','',
        '|Policy|Known ADE gain over floor (%)|Full-roster FDE|Own lower utility pp|Risk-supported record views|Known all/easy risk violations|Undefined selected easy|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in run.evaluation.NAMES:
        r=s['safety'][name];lines.append(f'|{name}|{interval(s["accuracy"][name])}|{interval(s["FDE"][name])}|{interval(s["floor_lower"][name])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|')
    lines+=['','|Policy|Switch rate|Neural row rate|Mean group cost-scale regret|Worst known easy degradation (%)|Easy views above2%|',
        '|---|---:|---:|---:|---:|---:|']
    for name,x in a.items():lines.append(f'|{name}|{cell(x["switch_rate"])}|{cell(x["neural_row_rate"])}|{cell(x["group_mean_normalized_regret"])}|{cell(x["worst_known_easy_degradation_percent"])}|{x["known_easy_views_above2"]}|')
    lines+=['','Rates pool repeated source/context/head views;regret is a mean of group means,not independent row evidence. Neural use is not a contribution ablation.',
        'Known net easy change and positive-harm ratios are separate. Predicted feasibility is not actual risk calibration.','',
        '|Subset known gain (%)|ADE raw|ADE guarded|TDEF raw|TDEF guarded|Frozen own joint|','|---|---:|---:|---:|---:|---:|']
    for subset in ('hard','easy'):
        x=s['subsets'][subset]['policies'];lines.append('|'+subset+'|'+'|'.join(cell(x[n]['ADE_gain_percent']) for n in (*run.evaluation.NAMES[:4],'frozen_own_joint'))+'|')
    lines+=['','## Compute And Verification','']
    for name,r in (('Real400update pilot',v['pilot']),('Formal training and full refits',t['freeze']),('Full72readout',v['complete']),('Separate new-score/action replay',v['replay'])):
        lines.append(f'{name}:PID{r["pid"]},{r["seconds"]:.6f}s,peakRSS{r["peak_RSS_bytes"]}B.')
    lines+=['','Native arm64CPU4/interop1/workers0,checkpoint50updates,heartbeat,explicit resume,phase lock. No speed-based reduction.',
        'Original full and record controls are exact. No new timed MILP is executed;new independent choices and frozen actions are exactly replayed.','',
        '## Fitting Gates','']+[f'- {k}: {"pass" if x else "fail"}' for k,x in s['fitting_gates'].items()]
    lines+=['',f'Fitting advancement:{s["fitting_advancement"]}. Independent confirmation:false. Deployment changed:false. Stage5C:false. SMC:false.','',
        '![Current-bank source comparison](figures/current_bank_deferral.svg)','',
        '## Limits','',
        'All149,510targets/74recordings/12localities/72groups remain.3,000paired locality bootstrap draws average repeated context and cost-seed views first.',
        'These SOURCE fitting intervals are not independent confirmation or three forecast pipelines. Full-roster FDE remains not_estimable where endpoints are unknown.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'No neural dynamics or physical-safety improvement is established by a cost-routing result.','',
        'Primary method: [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html).','']
    (p/'readout.md').write_text('\n'.join(lines))
    (p/'gates.md').write_text('# Current-Bank Fitting Gates\n\n'+'\n'.join(f'- {k}: {x}' for k,x in s['fitting_gates'].items())+
        f'\n\nFitting advancement:{s["fitting_advancement"]}. Independent confirmation:false. Deployment:false. Stage5C:false. SMC:false.\n')
    primary=s['effects']['guarded_tdef_minus_ade']['full'];r=s['safety']['tdef_guarded']
    (p/'failure_analysis.md').write_text('# Current-Bank Deferral: Measured Interpretation\n\n'+
        f'Primary known:{interval(primary["known_pp"])}pp;full lower:{interval(primary["lower_pp"])}pp.\n'+
        f'Guarded TDEF known gain:{interval(s["accuracy"]["tdef_guarded"])}%;guarded ADE:{interval(s["accuracy"]["ade_guarded"])}%.\n'+
        f'Required risk support:{r["supported_record_views"]}/{r["required_record_views"]};known all/easy violations:{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}.\n'+
        f'Worst guarded-TDEF known easy degradation:{cell(a["tdef_guarded"]["worst_known_easy_degradation_percent"])}%.\n\n'+
        'The loss control is internally matched;comparison to the earlier critic includes a different output architecture and initialization.\n'+
        'A useful observed mean does not resolve unknown-label bounds,undefined easy support or observed recording harm.\n'+
        'The frozen guard changes eligibility,not the underlying risk estimates;its failure would not prove that the score loss alone caused all risk errors.\n'+
        'Raw routing,guarded routing,within-loss guard effects and original controls are retained,including negative results.\n'+
        'Do not select a favorable seed,drop a locality,relax2%or open independent outcomes to rescue a SOURCE fit.\n'+
        'The published adapter is a strong-method control,not a new M3W contribution or evidence of a trajectory/scene/interaction increment.\n')
    (p/'model_data_card.md').write_text('# Model And Data Card\n\n'
        '36real Torch score heads;30,726active parameters each. Warm causal encoder,zero score/floor output initialization.\n'
        'One predicts expected normalized native partial ADE via softplus MSE;one uses the ICML2024Eq3continuous-cost logistic surrogate.\n'
        'Same seven current forecasts plus floor,462causal fields,anchor/envelope/context inputs and query coverage.\n'
        'Unavailable actions tie to floor. No future endpoint,goal,label mask or central velocity is an inference field.\n'
        'Raw independent selection and unchanged frozen own-critic risk-guarded selection. No new threshold fitting or physical guarantee.\n'
        'Exposed EuropeanSquares SOURCE,149,510targets,74recordings,12localities,obs8/pred12/rawstride12.\n'
        'Detector-silver/local-image,not metric,seconds,human gold,true3D or foundation. Three cost seeds share forecaster43.\n'
        'Raw data,features,caches,weights and third-party files stay outsideGit. No independent or whole-pipeline OOF claim.\n')
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),training_freeze_sha256=run.sha(p/'training_freeze.json'),
        complete_sha256=run.sha(p/'complete.json'),verification_sha256=run.sha(p/'verification.json'),archive_sha256=v['bundle']['archive_sha256'],
        new_real_fits=36,primary_optimizer_updates=208320,full_refit_updates=14336,fitting_advancement=s['fitting_advancement'],
        independent_confirmation=False,deployment_changed=False,submission_ready=False))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=v['summary'];names=('frozen_own_joint','ade_raw','ade_guarded','tdef_raw','tdef_guarded')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,name in enumerate(names):
        x=s['accuracy'][name];axes[0].bar(i,x['point'],color=('#555555','#326a95','#72a5c9','#9a6a16','#b85042')[i],width=.65)
        if x['CI95'] is not None:axes[0].vlines(i,*x['CI95'],color='black')
    axes[0].set_xticks(range(5),['Frozen own','ADE raw','ADE guard','TDEF raw','TDEF guard']);axes[0].set_ylabel('Known ADE gain over floor (%)')
    effect=s['effects']['guarded_tdef_minus_ade']['full']
    for i,key in enumerate(('known_pp','lower_pp')):
        x=effect[key]
        if x['CI95'] is not None:axes[1].hlines(i,*x['CI95'],color='#326a95');axes[1].plot(x['point'],i,'o',color='#326a95')
    axes[1].axvline(0,color='black',linewidth=.7);axes[1].set_yticks([0,1],['Known','Completion lower']);axes[1].set_ylim(-.3,1.3)
    axes[1].set_xlabel('Guarded TDEF minus guarded ADE utility (pp)')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE fitting only: matched current-bank published-loss control')
    (run.PUBLIC/'figures').mkdir(parents=True,exist_ok=True);run.PRIVATE.mkdir(parents=True,exist_ok=True)
    fig.savefig(run.PUBLIC/'figures/current_bank_deferral.svg');fig.savefig(run.PRIVATE/'current_bank_deferral.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    value=build();render(value);figure(value)
    print(json.dumps(dict(groups=72,fitting_advancement=value['summary']['fitting_advancement'],independent_confirmation=False,deployment_changed=False)))
