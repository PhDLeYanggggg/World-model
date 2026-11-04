"""Complete registered shared-context results; never select a deployment winner."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_moving_shared_harm_refinement as run
from scripts.report_m3w_moving_harm_refinement import factorization,text_once
from scripts.report_m3w_moving_amplitude_cost import interval,secondary_intervals
from scripts.report_m3w_harm_risk_taxonomy import category,LABELS

source=run.source
PUBLIC=run.PUBLIC


def aggregate_costs(values):
    actual=np.sum([r['realized_selected_known_moments']for r in values],0)
    predicted=np.sum([r['predicted_on_selected_known_moments']for r in values],0)
    return dict(selected=sum(r['selected']for r in values),selected_known=sum(r['selected_known']for r in values),
        **factorization(actual,predicted))


def matching_and_features(fits):
    rows=[];models={}
    for receipt in fits:
        state=run.api.base.read_checkpoint(source.checked(receipt['checkpoint']))
        models[receipt['job']['key'],receipt['arm']]=state
        importance=np.sum([tree.feature_importances_ for tree in state['trees']],axis=0)
        groupings=dict(local380=slice(0,380),current_cost5=slice(380,385),amplitude=slice(385,386),
            log_query_size=slice(386,387),shared48=slice(387,435))
        total=importance.sum()
        feature_nodes=np.concatenate([t.tree_.feature for t in state['trees']])
        shared_nodes=int((feature_nodes>=387).sum())
        if receipt['arm']=='local_zero'and shared_nodes:raise ValueError('Constant zero controls cannot split on shared fields')
        rows.append(dict(job=receipt['job']['key'],arm=receipt['arm'],tree_nodes=receipt['actual_tree_nodes'],
            shared_split_nodes=shared_nodes,importance_mass_by_group={k:float(importance[ix].sum()/total)
                if total>0 else None for k,ix in groupings.items()},
            impurity_diagnostic_not_causal_feature_contribution=True))
    jobs={r['job']['key']for r in fits}
    if len(fits)!=36 or len(jobs)!=18 or len(models)!=36:raise ValueError('All18fresh matched pairs required')
    for job in jobs:
        local,shared=(models[job,a]for a in run.api.ARMS)
        for key in ('identity','seed','settings','preprocess','frozen_prediction_sha256','fit_ids_sha256',
                'raw_feature_sha256','truth_sha256','legal_sha256','feature_dimension','feature_schema','legal_pairs','queries'):
            run.api.base.exact(local[key],shared[key])
    return dict(all18real_pairs_same_raw_inputs_targets_normalizer_seed_budget=True,rows=rows,
        learned_node_counts_not_parameter_exact=True)


def figures(summary,fits,taxonomy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pub=PUBLIC/'figures';preview=run.PRIVATE/'figure_previews'
    pub.mkdir(exist_ok=True);preview.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,metric,title in zip(axes,('known_pp','completion_lower_pp'),('Known labels','Completion lower')):
        for i,(name,row)in enumerate(summary['paired'].items()):
            v=row['joint_free'][metric]
            if v['point']is not None:
                ax.errorbar(v['point'],i,xerr=[[v['point']-v['CI95'][0]],[v['CI95'][1]-v['point']]],fmt='o',capsize=3)
        ax.axvline(0,color='.5',lw=1);ax.set_yticks(range(len(summary['paired'])),list(summary['paired']))
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Direct paired gain (pp), grouped95%CI')
    fig.suptitle('Registered development contrasts; not independent confirmation',fontsize=10)
    fig.tight_layout();fig.savefig(pub/'paired_effects.svg');fig.savefig(preview/'paired_effects.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3.3))
    primary=summary['paired']['shared_minus_local']['joint_free']
    for y,key,label in ((0,'known_pp','Known labels'),(1,'completion_lower_pp','Completion lower')):
        r=primary[key]
        ax.errorbar(r['point'],y,xerr=[[r['point']-r['CI95'][0]],[r['CI95'][1]-r['point']]],fmt='o',capsize=4)
        ax.annotate(f"{r['point']:+.4f} [{r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f}]pp",
            (r['point'],y),xytext=(0,14),textcoords='offset points',ha='center',fontsize=8)
    ax.axvline(0,color='.5',lw=1);ax.set_yticks([0,1],['Known labels','Completion lower'])
    ax.set_ylim(1.55,-.5);ax.set_xlabel('Shared minus fresh435wide local, grouped95% developmentCI (pp)')
    ax.set_title('Registered primary detail; all23supported exposed source groups')
    fig.tight_layout();fig.savefig(pub/'primary_detail.svg');fig.savefig(preview/'primary_detail.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3.6))
    for arm in run.api.ARMS:
        traces=[run.api.base.read_checkpoint(source.checked(r['checkpoint']))['trace']for r in fits if r['arm']==arm]
        x=np.asarray([[v['weighted_normalized_harm_MSE']for v in trace]for trace in traces])
        ax.plot(range(1,21),np.median(x,0),label=arm);ax.fill_between(range(1,21),x.min(0),x.max(0),alpha=.12)
    ax.set_yscale('log');ax.set_xlabel('Tree fitting round');ax.set_ylabel('Own-region weighted normalized H/EH MSE')
    ax.set_title('18fits per arm, median/min-max\nChoice weights differ: loss levels are not matched performance scores')
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(pub/'fitting_losses.svg');fig.savefig(preview/'fitting_losses.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));bottom=np.zeros(len(run.ARMS))
    colors=('#c6c6c6','#87b5d4','#c25857','#d6a153','#397f6b')
    for key,color in zip(LABELS,colors):
        values=np.array([taxonomy[a][key]for a in run.ARMS]);ax.bar(range(len(values)),values,bottom=bottom,label=key,color=color)
        for i,v in enumerate(values):
            if v>=18:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',fontsize=8)
        bottom+=values
    ax.set_xticks(range(len(run.ARMS)),run.ARMS,rotation=18,ha='right')
    ax.set_ylabel('Repeated recording views, not independent videos');ax.set_ylim(0,365)
    ax.set_title('All348views retained; priority partition, original2%contract unchanged')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.28,left=.09,right=.98,top=.87)
    fig.savefig(pub/'risk_footprint.svg');fig.savefig(preview/'risk_footprint.png',dpi=160);plt.close(fig)


def main():
    s=source.read(PUBLIC/'summary.json');v=source.read(PUBLIC/'verification.json');freeze=source.read(PUBLIC/'training_freeze.json')
    if (not v['all54fresh_cost_outputs_and_all108choice_scalar_blocks_exact']
            or v['summary_sha256']!=source.sha(PUBLIC/'summary.json')or freeze['complete_refiners']!=36):
        raise ValueError('Complete all fits, all held outcomes and exact replay before readout')
    fits=[source.read(source.checked(ref))for ref in freeze['fits']]
    for r in fits:source.checked(r['checkpoint'])
    matching=matching_and_features(fits)
    for arm in run.api.ARMS:
        if not source.read(PUBLIC/('full_refit_'+arm+'.json'))['exact']:raise ValueError('Both complete refits required')
    records=[row for ref in s['group_receipts']for row in source.read(source.checked(ref))['rows']]
    if len(records)!=138 or len({r['site']for r in records})!=23:raise ValueError('Complete supported-source readout required')
    c_cells=[cell for ref in freeze['common_region_receipts']for cell in source.read(source.checked(ref))['cells']]
    if len(c_cells)!=162:raise ValueError('All162common-region fitting cells required')
    fitting=[dict(region=region,cost_model=arm,**aggregate_costs([r['scores']for r in c_cells
        if r['region']==region and r['cost_model']==arm]))for region in run.FRESH for arm in run.FRESH]
    diagnostics=[source.read(source.checked(ref))for ref in s['cost_diagnostic_receipts']]
    if len(diagnostics)!=18 or not all(r['labels_used_after_all_decisions']for r in diagnostics):
        raise ValueError('Complete post-decision diagnostics required')
    held={arm:{policy:aggregate_costs([r['diagnostics'][arm][policy]['query_scores']for r in diagnostics])
        for policy in ('independent','frame_uniform','joint_matched','joint_free')}for arm in run.FRESH}
    sites=sorted({r['site']for r in records});usage=[];taxonomy={}
    for arm in run.ARMS:
        usage.append(dict(arm=arm,source_macro_switch_percent=100*float(np.mean([np.mean([
            r['policies'][arm]['joint_free']['switch_rate']for r in records if r['site']==site])for site in sites])),
            source_macro_amplitude_all_rows=float(np.mean([np.mean([r['usage'][arm]['mean_amplitude_all_rows']
                for r in records if r['site']==site])for site in sites])),
            empty_same_count_group_views=sum(r['matched_utility'][arm]['contrast']['known_pp']is None for r in records),
            required_group_views=138))
        taxonomy[arm]=dict.fromkeys(LABELS,0)
        for row in records:
            for record in row['policies'][arm]['joint_free']['recordings'].values():taxonomy[arm][category(record)]+=1
        if sum(taxonomy[arm].values())!=348:raise ValueError('All348views required')
    primary=s['paired']['shared_minus_local']['joint_free'];risk=s['safety']['shared_motion']['joint_free']
    lower_pass=primary['completion_lower_pp']['CI95']is not None and primary['completion_lower_pp']['CI95'][0]>0
    decision=dict(shared_context_robust_primary_positive=lower_pass,
        full_original_risk_pass=risk['full_original_risk_pass'],
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        submission_ready=False,new_deployment=False,new_neural_training=False,stage5c_executed=False,smc_enabled=False)
    execution={p:source.read(PUBLIC/(p+'_execution.json'))for p in ('pilot','train','evaluate','verify')}
    runtime=dict(phases=execution,retained_primary_trees=720,pilot_tree_fits=48,full_refit_tree_fits=40,
        uninterrupted_expected_tree_fit_calls=808,interrupted_extra_tree_fit_calls_not_inferred_from_this_total=True,
        private_artifact_bytes_excluding_figure_previews=sum(p.stat().st_size for p in run.PRIVATE.rglob('*')
            if p.is_file()and 'figure_previews'not in p.parts))
    for name,value in (('decision.json',decision),('runtime_summary.json',runtime),('fitting_summary.json',fitting),
            ('held_cost_summary.json',held),('policy_usage.json',usage),('risk_taxonomy.json',taxonomy),
            ('matching_and_feature_usage.json',matching),
            ('secondary_metrics.json',secondary_intervals(records,source.read(run.parent.CONFIG)))):
        source.once(PUBLIC/name,source.stable(value))
    fmt=lambda x:'undefined'if x is None else f'{x:.6f}'
    table=['|Joint-free contrast|Known effect pp [95%CI]|Direct completion-lower pp [95%CI]|','|---|---:|---:|']
    for name,row in s['paired'].items():table.append(f"|{name}|{interval(row['joint_free']['known_pp'])}|{interval(row['joint_free']['completion_lower_pp'])}|")
    risks=['|Cost/policy arm|Full support /348|Known all violations|Known easy violations|Undefined selected easy|Worst signed-net easy %|Switch %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        r=s['safety'][arm]['joint_free'];u=next(u for u in usage if u['arm']==arm)
        risks.append(f"|{arm}|{r['supported_record_views']}|{r['known_all_risk_violations']}|{r['known_easy_risk_violations']}|{r['undefined_easy_views']}|{fmt(r['worst_known_easy_net_degradation_percent'])}|{u['source_macro_switch_percent']:.5f}|")
    common=['|C common choice region|Cost model|H actual/pred|EH actual/pred|','|---|---|---:|---:|']
    for r in fitting:common.append(f"|{r['region']}|{r['cost_model']}|{fmt(r['moments']['H']['ratio'])}|{fmt(r['moments']['EH']['ratio'])}|")
    heldtable=['|H own joint choice, post-decision|H actual/pred|EH actual/pred|R actual/pred|ER actual/pred|','|---|---:|---:|---:|---:|']
    for arm in run.FRESH:heldtable.append('|'+arm+'|'+'|'.join(fmt(held[arm]['joint_free']['moments'][k]['ratio'])for k in ('H','EH','R','ER'))+'|')
    text_once(PUBLIC/'readout.md','# Shared past/current context in harm refiners\n\n'
        '36fresh435-column tree refiners /720primary trees,18cached-verified neural heads,3seeds. '
        'Both local-zero and shared controls are retrained before H.54fresh cost arrays and108choice scalar blocks replay exactly. '
        'Prior387decisions are cached-verified with fresh complete scalar replay; their costs are not freshly inferred.\n\n'
        +'\n'.join(table)+'\n\n'+'\n'.join(risks)+'\n\n'
        f'Registered robust primary passed: **{lower_pass}**. Full original risk passed: **{risk["full_original_risk_pass"]}**. '
        'No new deployment or submission readiness. All six policies and23sourcegroups remain in summary.json/groups. '
        'Original24primary stays NA; this is exposed development, not confirmation.\n\n'
        +'\n'.join(common)+'\n\n'+'\n'.join(heldtable)+'\n\n'
        'Within-region comparisons hold selection fixed. Own-region C loss/masses are not H safety. '
        'Pooled costs are not source-macro accuracy or recording-level certification. '
        'Empty same-count comparisons stay NA. Signed-net preservation cannot replace positive-harm/support.\n\n'
        'Resource phases and real durations are in runtime_summary.json. Saved choices are binding/feasibility verified where fresh costs exist; '
        'no bitwise MILP reoptimization is claimed. No new images, independent roles, metric/seconds, true3D, foundation, Stage5C or SMC.\n')
    m=held['shared_motion']['joint_free']['moments'];u=next(r for r in usage if r['arm']=='shared_motion')
    text_once(PUBLIC/'failure_analysis.md',f'''# What the shared-context test does and does not establish

## Fixed hypothesis and result

Only48already-admitted past/current group fields differ between two fresh435wide
controls. No future label or new goal construction enters inference. The old
negative shared neural study stays negative; a different cost function class
does not retrospectively repair it. This is not a new pooling architecture.

Known shared-minus-local joint effect: {interval(primary['known_pp'])}pp.
Direct completion lower: {interval(primary['completion_lower_pp'])}pp.
Robust primary pass: {lower_pass}. A nonpositive lower CI does not establish
that the unobserved true effect is negative; it fails the registered criterion.

## Safety and support remain separate

Shared known all/easy violations are{risk['known_all_risk_violations']}/
{risk['known_easy_risk_violations']}; full support{risk['supported_record_views']}/348.
Worst signed easy-net degradation is{fmt(risk['worst_known_easy_net_degradation_percent'])}%.
Undefined easy views{risk['undefined_easy_views']}; priority breakdown:
{source.json.dumps(taxonomy['shared_motion'],sort_keys=True)}.
No-change is analytic path identity, not an observed selected-risk certificate.
Raw overlapping violations are retained separately. Full risk passes only if
all required views meet the unchanged original contract.

Source-macro switch{u['source_macro_switch_percent']:.6f}%; empty same-count
contrasts{u['empty_same_count_group_views']}/138. Fewer switches can improve
risk without demonstrating useful intervention. These empty utility contrasts
are not silently dropped or converted to measured zero.

## Cost-estimation diagnosis

On H own joint selections, pooled shared actual/predicted H/EH are
{fmt(m['H']['ratio'])}/{fmt(m['EH']['ratio'])}; R/ER are
{fmt(m['R']['ratio'])}/{fmt(m['ER']['ratio'])}. See all162Cfixed-region cells,
all54Hdiagnostic cells and per-record metrics before attributing a unique cause.
C/H populations differ. Fitting magnitude repair does not prove held
calibration; neither pooled ratios nor distance reduction is a safety theorem.

Legacy crowd_mean_dx/dy fields are current centroid offsets, not shared camera
velocity. No verified ego-motion, image semantics, metric or raw-video time
calibration is available. Missing outcomes leave completion uncertainty.

## Next scientific action

Do not deploy or select an old cached control based on these held outcomes.
Retain same-region comparisons and identify record-level error/support tails
before another fit. If shared effects fail robustly, stop presenting these
aggregate48fields as the missing information. A richer admissible history or
independent risk-calibration study requires its own prospective definition.
Reserved20groups, DUT and DroneCrowd confirmation remain closed. SOURCE is
already-exposed development; no amount of replay makes it independent.
''')
    text_once(PUBLIC/'gate_readout.md',f'''# Gates, not a single completion score

|Condition|Result|
|---|---|
|Both435wide arms freshly trained, all36complete|pass|
|6real resume pilots and2complete first-job refits|pass|
|All54fresh cost outputs and108choice scalar blocks exact replay|pass|
|Original387cached policies match all prior scalar blocks|pass; old costs not freshly inferred|
|C162common-region and H54post-decision diagnostics|complete|
|Registered shared-minus-local completion-lower CI strictly positive|{lower_pass}|
|Shared full original348view2%contract|{risk['full_original_risk_pass']}|
|Original24group primary|not_estimable, not passed|
|Independent confirmation|not_run; reserved roles closed|
|Deployment / submission ready|false / false|
|Stage5C / SMC|false / false|

Engineering completion does not imply a scientific pass. Signed-net, positive
harm, support and utility remain distinct. Three training seeds and3000grouped
bootstrap are development statistics, not a new independent test.
''')
    text_once(PUBLIC/'model_data_card.md','''# Component and data card

This component estimates upward H/EH cost corrections for fixed trajectory
candidates. It is not a neural dynamics model or new forecaster.36new models
contain20small trees each; frozen neural cost controls number18. Both input
widths are435, not necessarily identical learned node counts. R/ER are
pointwise unchanged. B can shrink for causal-envelope coherence.

Inputs: local380causal fields, current predictedcost5, amplitude, logquerysize,
and either48zeros or original-C-standardized past/current group fields. No
scene/source identity, future endpoint, future validity/easy membership, central
velocity or test-derived goals enter inference. Labels enter Csupervision and
post-decision Hevaluation only. No new normalization or data role is fitted.

Same64admitted SOURCE recordings,195286past-eligible windows.58recordings in
23supported source groups support native8/12annotations; original24primary is
not_estimable. Grouping is conservative, not verified IID physical sites.
Partial futures use available-label ADE, not full12step trajectory ADE. Pixel
coordinates, video FPS/geometry unverified. No seconds, metric, true3D,
foundation or human-gold claim. Reserved20Moving groups, DUT/DroneCrowd closed.

Nativearm64CPU4/interop1/workers0; checkpoints per tree, event heartbeat and
exclusive lock.6real2+2versus4resume cells,2complete full refits and complete
saved-ledger replay. Prior387controls are cached choices with current input
hashes and fresh exact scalar replay, not new cost inference/feasibility.
Nonoptimal solver outcomes stay visible; no bitwise optimization replay claim.

Use only for exposed-development research. See readout, all per-record risks,
empty controls and failure analysis. No new deployment, Stage5C or SMC.
''')
    per=['# Every supported SOURCE group','', '|Source|Shared-local known pp|Shared-local lower pp|Shared-local upper pp|', '|---|---:|---:|---:|']
    for site in sites:
        per.append('|'+site+'|'+'|'.join(f"{primary[k]['by_source_group'][site]:+.6f}"for k in
            ('known_pp','completion_lower_pp','completion_upper_pp'))+'|')
    per+=['','All23groups retained, after producer/seed averaging. Missing-label uncertainty and negative sources are not dropped. Not independent physical scenes.']
    text_once(PUBLIC/'per_source.md','\n'.join(per)+'\n')
    text_once(PUBLIC/'operation_zh.md','''# 本轮复现说明

本轮是冻结神经预测器上的树代价估计实验，不是新训神经世界动力学。
协议、代码、配置和registration先提交，再拟合。使用原生arm64环境。

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_shared_harm_refinement.py
```

上述命令是阶段入口。已经完整结束的阶段不会覆盖旧结果；未完成阶段加
`--resume`，继续原定义、原预算。不要重命名结果或通过重跑优化器挑测试赢家。
每棵树保存checkpoint，私有events.jsonl和heartbeat.json记录PID、耗时、进度。
num_workers=0，CPU4/interop1，不用资源探测或多进程数据加载。

两组均435维；差异仅为末48维保留还是置零。旧387维对照只读历史决策缓存，
校验SHA和逐行输入绑定，再重新计算所有指标；不是新推理旧模型代价。
完整verify重新推理54组代价并核对108组决策的指标，不声称MILP位级重求解。

结果看readout.md和gate_readout.md。负结果、不可估和空同切换数对照都保留。
23组是已经暴露的开发证据，原24组primary仍不可估，未启用独立校准或确认。
没有metric/seconds/true3D/foundation声明；Stage5C和SMC始终关闭。
''')
    text_once(PUBLIC/'next_steps.md','''# Next actions

1. Interpret shared versus fresh equal-width local as the primary feature test;
   do not pick a cached arm or a favorable source after held readout.
2. Use the same-selection C/H diagnostics and record-level support taxonomy to
   distinguish fitting bias, residual held errors and unobserved outcomes.
3. Preserve closed independent roles. New admissions/role decisions require
   approval before outcomes; exposed development replay cannot certify safety.

No new neural world-model claim, deployment, Stage5C or SMC is authorized.
''')
    figures(s,fits,taxonomy)
    print(source.json.dumps(dict(decision=decision,primary=primary['completion_lower_pp'],risk=risk),sort_keys=True))


if __name__=='__main__':main()
