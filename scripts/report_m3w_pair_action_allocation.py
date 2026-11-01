"""Export a complete fixed-design interaction control,including negative results."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_pair_action_allocation as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def build():
    reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Frozen scientific definition changed')
    s, complete, replay = [json.loads((run.PUBLIC/name).read_text()) for name in
        ('summary.json','complete.json','verification.json')]
    if (not replay['all72groups_exact'] or not replay['all72_old_controls_exact']
            or complete['summary_sha256'] != run.sha(run.PUBLIC/'summary.json')
            or replay['summary_sha256'] != run.sha(run.PUBLIC/'summary.json')
            or replay['complete_sha256'] != run.sha(run.PUBLIC/'complete.json')
            or complete['registration_sha256'] != run.sha(run.REG)):
        raise ValueError('Complete separate replay and original controls required')
    bundle = verify(ROOT,run.PUBLIC); groups = aggregate_groups(run.NAME); original = aggregate_groups(run.unit.NAME)
    rows = [groups[Path(r['path']).name] for r in complete['groups']]
    rebuilt = run.evaluation.summarize(rows,reg['contexts'],json.loads(run.CONFIG.read_text()))
    records = len({(r['source'],rec) for r in rows for rec in r['result']['recordings']})
    rebuilt.update(unique_target_rows=149510,unique_recordings=records)
    if rebuilt != s or records != 74 or len(groups) != 72: raise ValueError('Complete canonical cohort required')
    for name, row in groups.items():
        old = original[name]
        if (row['prediction_sha256'] != old['prediction_sha256']['selected_set']
                or row['ids_sha256'] != old['ids_sha256']
                or row['action_sha256']['unary_joint'] != old['action_sha256']['set_joint']
                or row['result']['policies']['unary_joint'] != old['result']['policies']['set_joint']):
            raise ValueError('Original controls changed')
    counts=[r['result']['allocation'] for r in rows]; edges=sum(v['graph_edges'] for v in counts)
    graph_description=dict(edge_views=edges,zero_edge_groups=sum(v['graph_edges']==0 for v in counts),
        pooled_before=sum(v['graph_pair_mass_before'] for v in counts)/edges if edges else None,
        pooled_after=sum(v['graph_pair_mass_after'] for v in counts)/edges if edges else None,
        full_locality_CI=s['graph_pair_change'],supplementary_pooled_point_not_gate_replacement=True)
    pairless_unsupported=[name for name,old in original.items() if old['result']['pair_proxy']['pair_requested_step_views']==0
        and not old['result']['policies']['set_joint']['policy']['budget_status']['supported']]
    choice_counts={a:{str(k):sum(r['result']['policies'][a]['action_counts'][str(k)] for r in rows)
        for k in range(-1,7)} for a in run.api.NAMES}
    mass_changes={}
    for r in rows:
        result=r['result']; ref=result['contrasts']['full']['full_reference_mass_counted_once']
        if ref<=0: raise ValueError('Defined original reference required')
        mass_changes.setdefault(r['source'],[]).append([100*(result['policies']['pair_joint']['policy'][k]
            -result['policies']['unary_joint']['policy'][k])/ref for k in ('selected_known_benefit_mass','selected_known_harm_mass')])
    per_source={k:run.np.mean(v,0).tolist() for k,v in mass_changes.items()}
    if any(abs(v[0]-v[1]-s['effects']['known_pp']['by_locality'][k])>1e-8 for k,v in per_source.items()):
        raise ValueError('Known benefit/harm decomposition identity changed')
    decomposition=dict(benefit_delta_pp=float(run.np.mean([v[0] for v in per_source.values()])),
        harm_delta_pp=float(run.np.mean([v[1] for v in per_source.values()])),
        unknown_disagreement_penalty_pp=s['effects']['known_pp']['point']-s['effects']['lower_pp']['point'],
        positive_known_sources=sum(v>0 for v in s['effects']['known_pp']['by_locality'].values()),
        unchanged_known_sources=sum(v==0 for v in s['effects']['known_pp']['by_locality'].values()),
        source_count=12,by_source_benefit_harm=per_source,posthoc_descriptive_not_causal_feature_diagnosis=True)
    return dict(summary=s,complete=complete,replay=replay,bundle=bundle,rows=rows,graph_description=graph_description,
        choice_counts=choice_counts,decomposition=decomposition,
        unchanged_pairless_unsupported_controls=sorted(pairless_unsupported),
        pilot=json.loads((run.PUBLIC/'pilot.json').read_text()),
        verification_runtime=json.loads((run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    if v['CI95'] is None: return 'not_estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def render(v):
    s = v['summary']; risk = s['safety']['pair_joint']; complete = v['complete']; replay = v['verification_runtime']
    primary = s['effects']['lower_pp']; graph = s['graph_pair_change']
    lines = ['# Predicted-Path Interaction: Full Real-Data Control','',
        '## Evidence Source','',
        'fresh_run: complete seven-action pair-aware allocation,full original outcome readout and a separate complete replay.',
        'cached_verified: 18 neural cost heads,all candidate/floor forecasts,past-only features,original unary predictions and full-policy choices.',
        'not_run: new neural or trajectory training,independent calibration/confirmation,and SDD deployment retesting.',
        'This is exposed SOURCE fitting resubstitution,not whole-pipeline OOF or independent generalization. No deployment changes.','',
        '## Fixed Factor','',
        'The original query allocator shares unary gain and all/easy risk budgets but does not optimize pairwise trajectories.',
        'This control adds one fixed predicted-path contraction cost over a past-position four-nearest-neighbor graph.',
        'Every query retains exactly the original intervention count,action availability and original two-percent predicted-risk constraints.',
        'The objective is negative total predicted net gain divided by total past query scale,plus the mean graph cost(weight1.0).',
        'Candidate/floor paths are restored to common local image coordinates. No future observations enter this decoder.',
        'The geometric proxy is not a body-aware collision metric,physical safety guarantee or method novelty claim.','',
        '## Registered Results','',
        '| Pair minus unary | Percentage points [conditional95% CI] |','|---|---:|']
    for k in ('known_pp','lower_pp','upper_pp'): lines.append(f'| {k} | {fmt(s["effects"][k])} |')
    lines += ['', '| Policy | Known ADE gain over floor (%) | Original recording risk supported | Known easy-risk violations | Undefined easy |',
        '|---|---:|---:|---:|---:|']
    for a in run.api.NAMES:
        r = s['safety'][a]
        lines.append(f'| {a} | {fmt(s["accuracy"][a])} | {r["supported_record_views"]}/{r["required_record_views"]} | {r["known_easy_risk_violations"]} | {r["undefined_easy_record_views"]} |')
    lines += ['',f'Full-locality graph contraction change: {fmt(graph)}. Negative is less predicted contraction,not lower realized error.',
        f'Supplementary pooled graph points: {v["graph_description"]["pooled_before"]} before,{v["graph_description"]["pooled_after"]} after.',
        f'Zero-edge groups: {v["graph_description"]["zero_edge_groups"]};their missing graph statistic is not imputed as zero or removed from the utility/risk cohort.',
        f'Changed query views: {s["changed_queries"]}/{s["query_views"]}; solver calls: {s["solver_calls"]}; nonoptimal solver returns: {s["nonoptimal_solver_queries"]}.',
        'Every timeout or failed numerical postcheck retains the original parent and remains in the full scoring roster. No cohort-wide optimality claim.',
        'Unavailable requested FDE endpoints remain missing; known partial endpoint evidence cannot certify full-roster FDE.',
        '![Full control](figures/policy_readout.svg)','',
        '## Interpretation','',
        f'The registered primary completion-lower difference is {fmt(primary)}.',
        f'The original full-risk requirement is supported in {risk["supported_record_views"]}/{risk["required_record_views"]} recording/context/seed views.',
        f'Fitting advancement: {str(s["fitting_advancement_gate"]).lower()}. This alone cannot authorize deployment or independent confirmation.',
        'Lower geometric contraction must not be substituted for ADE improvement or original positive-harm safety.',
        f'{len(v["unchanged_pairless_unsupported_controls"])} old zero-pair groups are already risk-unsupported and retain their exact unary choices.',
        'This pair-only control therefore cannot certify complete original risk,even if crowded-query utility improves. It does not repair those unary failures.',
        'Known-label utility,completion-lower utility and individual-policy lower bounds have different estimands;their differences are not interchangeable.','',
        '## Descriptive Accounting','',
        f'Known benefit change: {v["decomposition"]["benefit_delta_pp"]:+.6f}pp;known harm change: {v["decomposition"]["harm_delta_pp"]:+.6f}pp.',
        'Benefit change minus harm change exactly reproduces the paired known utility difference in every source.',
        f'Positive known source effects: {v["decomposition"]["positive_known_sources"]}/12;unchanged sources: {v["decomposition"]["unchanged_known_sources"]}/12.',
        f'Neural action6 views: {v["choice_counts"]["unary_joint"]["6"]} -> {v["choice_counts"]["pair_joint"]["6"]}. Counts repeat contexts/critic seeds,not distinct people.',
        'These describe changed assignments,not a causal identification of missing intent,detector noise or feature insufficiency.','',
        '## Population And Statistics','',
        'All149,510 unique targets,74 recordings,72 original source/context/critic-seed groups and12 exposed source localities remain.',
        'Bootstrap3,000 paired locality resamples after averaging repeated contexts and critic seeds. Overlapping windows are not independent samples.',
        'Critic seeds17/29/43 share forecast-producer seed43;this is not a three-seed forecaster experiment.',
        'Per-source effects appear in locality_effects.csv. The graph includes query targets only,not all visible agents.','',
        '## Runtime And Recovery','',
        f'Pilot PID{v["pilot"]["pid"]}: exact interrupted recovery;18 queries/402 rows,peakRSS{v["pilot"]["peak_RSS_bytes"]}B.',
        f'Full readout PID{complete["pid"]}: {complete["seconds"]:.6f}s from runtime_ready,peakRSS{complete["peak_RSS_bytes"]}B.',
        f'Separate complete redecoding/replay PID{replay["pid"]}: {replay["seconds"]:.6f}s,peakRSS{replay["peak_RSS_bytes"]}B.',
        'Initial admission/loading is outside these phase timers. Native arm64 CPU4,interop1,workers0;no CREATE job submitted.',
        'Config,registration,heartbeat,progress checkpoints and exact frozen controls are retained. Weights/private rows are not in Git.','',
        '## Scope','',
        'EuropeanSquares oblique detector-silver local image coordinates;obs8/pred12,stride12 raw frames.',
        'Not SDDt50,metric,seconds-level,human gold,true3D or a foundation world model.',
        'Independent roles and reserved DroneCrowd labels remain closed. Stage5C and SMC remain disabled.','']
    return '\n'.join(lines)


def export(v):
    s = v['summary']; public = run.PUBLIC
    (public/'readout.md').write_text(render(v))
    with (public/'locality_effects.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(('source','known_pp','completion_lower_pp','graph_pair_change'))
        for source in sorted(s['effects']['known_pp']['by_locality']):
            writer.writerow((source,s['effects']['known_pp']['by_locality'][source],
                s['effects']['lower_pp']['by_locality'][source],s['graph_pair_change'].get('by_locality',{}).get(source)))
    risk=s['safety']['pair_joint']; primary=s['effects']['lower_pp']; graph=s['graph_pair_change']
    failure=['# Failure Analysis','',
        'This registered experiment isolates predicted-path interaction in the decoder,not missing features or a new neural architecture.',
        f'Primary robust utility: {fmt(primary)}. Sparse-graph contraction change: {fmt(graph)}.',
        f'Original recording-risk support: {risk["supported_record_views"]}/{risk["required_record_views"]}.',
        'If geometry improves without utility or safety,this control rejects this specific fixed pair penalty as a sufficient repair.',
        'It does not establish that interaction is useless:the proxy lacks physical scale/body extent and only covers current query targets.',
        'A fixed coefficient may trade unary forecast benefit for a proxy with different semantics. It was not tuned on these outcomes.',
        'Current costs remain fitted on exposed SOURCE data;neither this result nor a replay is independent confirmation.',
        'Do not respond by tuning the same outcome table or redefining the original two-percent risk. Retain the frozen forecasts and deployment.',
        'Next isolate whether predicted contraction corresponds to actual relative prediction harm using training-only labels,',
        'then consider a learned interaction cost only if that diagnostic supports predictive value. Keep negative results and unknown rows.','']
    (public/'failure_analysis.md').write_text('\n'.join(failure))
    (public/'model_data_card.md').write_text('\n'.join(['# Model And Data Card','',
        'No new model weights. Frozen18 selected-set critics with462 causal fields and original seven-action/floor forecasts.',
        'New deterministic decoder: one-hot action assignment and pair-transport MILP;fixed original per-query count and risk budgets.',
        'Inputs: predicted moments,past-only availability/history scales,current origins,relative rollouts,and row/query keys.',
        'No target endpoint,future validity mask,evaluation easy flag,central velocity or test endpoint goals are inference inputs.',
        'Data: all149510 original targets/74 recordings/12 exposed EuropeanSquares localities;detector-silver pixels,obs8/pred12/rawstride12.',
        'Not all-visible-agent forecasting,independent calibration,a physical-safety certificate or deployable world dynamics.',
        'Three critic seeds are not three forecasting seeds. Requested endpoint FDE support is incomplete.',
        'No new data permission,external upload,Stage5C execution,SMC,metric or seconds-level claim.','']))
    (public/'reproduction_zh.md').write_text('\n'.join(['# 复现说明','',
        '本轮是冻结预测器后的真实联合决策实验，没有重新训练神经轨迹模型。公开仓库不包含数据、缓存或权重；需要本机已有且授权的数据。',
        '使用原生 arm64 `.venv-pytorch/bin/python`；CPU 4 线程、interop 1、DataLoader workers 0。',
        '首次运行顺序：','```bash',
        '.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py register',
        '.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py pilot',
        '.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py eval',
        '.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_pair_action_allocation.py',
        '.venv-pytorch/bin/python -m pytest tests/test_m3w_pair_action_allocation.py tests/test_m3w_pair_action_safety.py -q',
        '```',
        '已完成的结果不可覆盖。中断后用 `eval --resume`；只有确认进程已终止才恢复，不能重复启动。',
        f'完整 readout PID：{v["complete"]["pid"]}；复核 PID：{v["verification_runtime"]["pid"]}。',
        '日志、心跳与断点在 data/stage_cvpr2027_experiments/european_pair_action_allocation_v1/；不会进入 Git。',
        '两秒是每个 MILP 的求解预算，不是中止慢实验的总时限。求解失败保留原策略并计入所有评价。',
        '结果只属于已暴露 SOURCE 拟合数据。不要称独立验证、米/秒预测、人工 gold、true 3D 或 foundation。','']))
    (public/'paper_results_component.md').write_text('\n'.join(['# Predicted-Path Coupling Control','',
        'We keep all forecast and risk-estimation weights fixed and add a causal predicted-path interaction penalty to seven-action joint assignment.',
        'The assignment preserves the original per-query intervention count and all/easy risk constraints;nonoptimal solves retain the original action set.',
        f'The paired completion-lower utility difference is {fmt(primary)} percentage points,with a graph contraction change of {fmt(graph)}.',
        f'The original recording-risk contract is supported in {risk["supported_record_views"]}/{risk["required_record_views"]} repeated views.',
        'Intervals condition on twelve exposed fitting localities and are not independent confirmation. This control cannot establish a physical-safety or novel world-model claim.',
        'This is one results component,not a complete paper or submission-ready package.','']))
    gates=['# Interaction-Control Gates','',f'Fitting advancement: {s["fitting_advancement_gate"]}','']
    gates += [f'- {k}: {v}' for k,v in s['fitting_gates'].items()]
    gates += ['', 'Independent confirmation: false','Submission ready: false','Stage5C executed: false','SMC enabled: false','']
    (public/'gates.md').write_text('\n'.join(gates))
    (public/'next_steps.md').write_text('\n'.join(['# Next Steps','',
        '1. Preserve this fixed decoder result and the original deployment;do not tune its coefficient against the same outcomes.',
        '2. Test training-only correspondence between the predicted interaction proxy and actual forecast harm before adding learned coupling.',
        '3. Only after a fitting repair passes original risk,freeze the full pipeline for independent calibration/confirmation and three forecasting seeds.','']))
    run.write(public/'graph_description.json',v['graph_description'])
    run.write(public/'choice_description.json',v['choice_counts'])
    run.write(public/'descriptive_decomposition.json',v['decomposition'])
    run.write(public/'evidence.json',dict(summary_sha256=run.sha(public/'summary.json'),
        complete_sha256=run.sha(public/'complete.json'),verification_sha256=run.sha(public/'verification.json'),
        bundle_sha256=run.sha(public/'metric_groups.zip'),full_replay_exact=True,new_training_updates=0,
        full_repo_tests_rerun=False,independent_confirmation=False,submission_ready=False))
    figure(s)


def figure(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    np=run.np; plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':run.NAME})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    vals=[s['accuracy'][a] for a in run.api.NAMES]
    axes[0].bar([0,1],[v['point'] for v in vals],color=['#557D86','#AF3D3D'])
    axes[0].errorbar([0,1],[v['point'] for v in vals],yerr=np.array([[v['point']-v['CI95'][0],v['CI95'][1]-v['point']] for v in vals]).T,fmt='none',color='black',capsize=4)
    axes[0].set_xticks([0,1],['Frozen unary','Pair-aware']); axes[0].set_ylabel('Known ADE gain over floor (%)')
    vals=[s['effects']['known_pp'],s['effects']['lower_pp']]
    axes[1].axvline(0,color='#777777',lw=1)
    for i,v in enumerate(vals):
        axes[1].errorbar(v['point'],i,xerr=[[v['point']-v['CI95'][0]],[v['CI95'][1]-v['point']]],fmt='o',color='#AF3D3D',capsize=4)
    axes[1].set_yticks([0,1],['Known difference','Completion lower']); axes[1].set_xlabel('Pair minus unary (percentage points)')
    for ax in axes: ax.grid(alpha=.15)
    fig.suptitle('Exposed SOURCE fitting control;conditional locality intervals,not confirmation')
    fig.tight_layout(rect=(0,0,1,.91)); (run.PUBLIC/'figures').mkdir(exist_ok=True)
    svg=run.PUBLIC/'figures/policy_readout.svg'; fig.savefig(svg,bbox_inches='tight',metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'policy_readout.png',dpi=160,bbox_inches='tight'); plt.close(fig)


if __name__=='__main__':
    value=build(); export(value)
    print(json.dumps(dict(groups=value['summary']['groups'],full_replay_exact=True,
        fitting_advancement=value['summary']['fitting_advancement_gate'],submission_ready=False)))
