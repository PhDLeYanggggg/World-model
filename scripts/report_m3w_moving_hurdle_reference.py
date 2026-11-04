"""Prediction-study readout; no reference improvement is called policy safety."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_moving_hurdle_reference as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval

source=run.source;PUBLIC=run.PUBLIC


def main():
    freeze=source.read(PUBLIC/'training_freeze.json');ev=source.read(PUBLIC/'evaluation.json');verified=source.read(PUBLIC/'verification.json')
    oof=source.read(PUBLIC/'oof_verification/verification.json')
    if (freeze['complete_jobs']!=18 or ev['complete_jobs']!=18
            or not verified['all18held_reference_prediction_arrays_and_metrics_exact']
            or verified['evaluation_sha256']!=source.sha(PUBLIC/'evaluation.json')
            or not oof['all18OOF_metrics_and_model_choices_exact']):raise ValueError('Complete all training, OOF and H replay first')
    fits=[source.read(source.checked(ref))for ref in freeze['fits']]
    for row in fits:
        for model in row['models']:source.checked(model['checkpoint'])
    for task in run.api.TASKS:
        if not source.read(PUBLIC/('full_refit_'+task+'.json'))['exact']:raise ValueError('All full refits required')
    held=[source.read(source.checked(ref))for ref in ev['receipts']]
    records=[dict(site=site,producer_fold=row['job']['producer_fold'],seed=row['job']['seed'],
        metrics=metrics,selected=row['selected_by_C_OOF'])for row in held for site,metrics in row['by_source'].items()]
    if len(records)!=138 or len({r['site']for r in records})!=23:raise ValueError('All supported SOURCE groups required')
    cfg=source.read(run.lineage.parent.CONFIG);sites=sorted({r['site']for r in records})
    ci=lambda getter:source.group_interval(records,getter,sites,cfg)
    contrasts={
        'direct_minus_prior_MSE_reduction':lambda r:r['metrics']['prior']['MSE']-r['metrics']['direct']['MSE'],
        'hurdle_minus_prior_MSE_reduction':lambda r:r['metrics']['prior']['MSE']-r['metrics']['hurdle']['MSE'],
        'hurdle_minus_direct_MSE_reduction':lambda r:r['metrics']['direct']['MSE']-r['metrics']['hurdle']['MSE'],
        'C_selected_minus_prior_MSE_reduction':lambda r:r['metrics']['prior']['MSE']-r['metrics'][r['selected']]['MSE'],
        'event_minus_prior_Brier_reduction':lambda r:r['metrics']['event']['prior_Brier']-r['metrics']['event']['Brier']}
    effects={name:ci(fn)for name,fn in contrasts.items()}
    absolute={name:ci(lambda r,name=name:r['metrics'][r['selected']if name=='C_selected'else name]['MSE'])
        for name in ('prior','direct','hurdle','C_selected')}
    event={name:ci(lambda r,name=name:r['metrics']['event'][name])for name in ('Brier','prior_Brier','positive_rate','AUROC','AUPRC','log_loss')}
    selected={name:sum(r['selected_by_C_OOF']==name for r in fits)for name in ('prior','direct','hurdle')}
    mean_effect=effects['C_selected_minus_prior_MSE_reduction'];hurdle_effect=effects['hurdle_minus_direct_MSE_reduction']
    positive=lambda r:r['CI95']is not None and r['CI95'][0]>0
    decision=dict(C_selected_reference_prediction_lift=positive(mean_effect),hurdle_over_direct_lift=positive(hurdle_effect),
        event_Brier_lift=positive(effects['event_minus_prior_Brier_reduction']),reference_prediction_only=True,
        new_decisions=False,trajectory_lift_tested=False,record_risk_calibration_proved=False,
        original_full_risk_repaired=False,new_deployment=False,submission_ready=False,
        independent_confirmation=False,stage5c_executed=False,smc_enabled=False)
    runtime={p:source.read(PUBLIC/(p+'_execution.json'))for p in ('pilot','train','evaluate','verify')}
    runtime['OOF_verification']=source.read(PUBLIC/'oof_verification/execution.json')
    for name,value in (('summary.json',dict(effects=effects,absolute_MSE=absolute,event=event,selection_counts=selected,
            supported_group_views=138,supported_source_groups=23,original24group_primary='not_estimable_not_passed')),
            ('decision.json',decision),('runtime_summary.json',runtime)):
        source.once(PUBLIC/name,source.stable(value))
    table=['|Fixed H contrast; positive is lower loss|Normalized loss reduction [grouped95%CI]|','|---|---:|']
    for name,row in effects.items():table.append(f'|{name}|{interval(row)}|')
    text_once(PUBLIC/'readout.md','# Source-excluded reference prediction\n\n'
        'Fresh216forests /9216primary trees, three actual seeds, direct64versus hurdle32+32trees. '
        'No trajectory model, policy or threshold changes. All108OOF arrays and216fit bindings replay; '
        'all18held prediction sets (108arrays) and metrics replay exactly.\n\n'
        +'\n'.join(table)+'\n\n'+f'C-only source-OOF selections:{source.json.dumps(selected,sort_keys=True)}. '
        f'Positive C-selected reference prediction effect:{decision["C_selected_reference_prediction_lift"]}; '
        f'positive hurdle-versus-direct effect:{decision["hurdle_over_direct_lift"]}; '
        f'positive event Brier effect:{decision["event_Brier_lift"]}.\n\n'
        'The intervals concern normalized ER prediction MSE or event Brier, NOT ADE/FDE percentages, '
        'trajectory utility, physical safety or independent risk calibration. All23supported source groups '
        'are exposed development; original24primary stays not_estimable. Null source AUCs remain not_estimable. '
        'Point prediction can improve while a finite selection still has no supporting event. '
        'No conditional independence of agents, no record-event guarantee and no missing-at-random assumption.\n\n'
        'Unknown-label rows remain in inference but outside this supervised conditional-known-label loss; '
        'this head cannot simply replace the original conservative bounded-reference decoder without '
        'accounting for unknown label availability. Positive conditional heads use the original loss measure '
        'restricted to positive events, not a second source/query rebalance.\n\n'
        'All model selection is complete source-excluded internal C validation. Frozen producer forecasts '
        'exclude every C/H source; no C-trained critic or global C normalizer is an input. '
        'Inputs are raw local380past/current/predicted-rollout descriptors, logquerysize and logcausalscale. '
        'Future masks and endpoints are labels only. No new official data roles or reserved-source opening.\n\n'
        'Private weights/OOF arrays are not committed. See per-source results, exact provenance verification, '
        'runtime receipts and operation guide. No metric/seconds/true3D/foundation/human-gold claim, Stage5C or SMC.\n')
    lines=['# Every supported source, no held winner selection','',
        '|Source|Selected-minus-prior MSE reduction|Hurdle-minus-direct MSE reduction|Event Brier reduction|','|---|---:|---:|---:|']
    for site in sites:
        lines.append('|'+site+'|'+'|'.join(f"{effects[k]['by_source_group'][site]:+.8f}"for k in
            ('C_selected_minus_prior_MSE_reduction','hurdle_minus_direct_MSE_reduction','event_minus_prior_Brier_reduction'))+'|')
    text_once(PUBLIC/'per_source.md','\n'.join(lines)+'\n')
    text_once(PUBLIC/'model_data_card.md','''# Reference-head model/data card

Prediction component only. Direct ExtraTrees regression uses64trees; a two-part
comparison uses32event-classification and32conditional-regression trees. All
depth6/minleaf64/maxfeatures32, bootstrapFalse, fit_threads1. Learned node counts
may differ despite equal tree budgets. No novel architecture is claimed.

382inputs: raw local380 descriptors, logcurrentquerysize, logcausalhistoryscale.
Predicted rollouts are producer-excluded; no learned critic output, global
Cscaler, source ID, future truth/mask/easy label or endpoint-derived test goals.
Inner source3fold fitting excludes entire validation sources for every learned
component. No new permanent official roles. Source/known-query weights are
conditioned, not rebalanced, for the positive-reference amount head.

Same195286past-eligible SOURCE rows from64admitted recordings. Predictions cover
the full role-specific roster; unknown futures are not negative event targets.
The estimand is ER>0 and available-label normalized ER, conditional on supplied
supervision; partial12step labels are not completed. It is not physical future
support or an independent record-event probability. No MAR/agent-independence
assumption justifies transporting it to unknown outcomes or finite-record risk.

Source23supported groups are exposed development, not verified IID physical
sites or independent confirmation. Original24primary remains not_estimable.
Reserved20Moving groups, DUT calibration and DroneCrowd confirmation stay closed.
No image/simulation admission, human-gold promotion, metric/homography/seconds,
true3D or foundation claim. Source-time provenance remains limited.

Nativearm64CPU4/interop1/workers0 with sequential forests, checkpoint every8trees,
PID/event heartbeat, complete source/feature/target/weight bindings.9real resume
cells and3full first-job refits;216model bindings/108OOF arrays and108Harrays
replay. No new decision is made, so this study cannot repair trajectory or
original2%deployment gates by itself. Stage5C/SMC off.
''')
    text_once(PUBLIC/'failure_analysis.md',f'''# Prediction evidence is not a repaired policy

Selected-reference prediction reduction: {interval(mean_effect)} in normalized
MSE units. Hurdle minus direct reduction: {interval(hurdle_effect)}.
Event Brier reduction: {interval(effects['event_minus_prior_Brier_reduction'])}.
Every source remains in per_source.md. Source-OOF choices were{selected}.

Whether positive or negative, these results concern a missing reference target,
not an executed controller. A conditional mean or event probability does not
guarantee realized supporting reference in a finite selected record. The
preceding51active unsupported views and63all-risk false-feasible views are not
removed or recertified. ADE/FDE and all existing decisions remain unchanged.

OOF prior comparisons are properly inside train folds, no globally Ctrained
critic/scaler inputs, unknown labels excluded from supervised loss. If a head
improves, test it next in a separately frozen causal admission study with the
full original2%risk and same-count utility controls. Do not multiply independent
agent probabilities or inject future visibility to make support appear certain.
If the hurdle does not beat direct regression, retain the simple comparator;
factorization alone is not the method contribution. No Hretuning is allowed.

The official source roles, missing-future limitations, dataset-local pixels and
native8/12steps remain. Independent final evidence, support calibration and
trajectory gains are still missing. Not a submission-ready/foundation/true3D
model. No Stage5C or SMC.
''')
    text_once(PUBLIC/'operation_zh.md','''# 本轮参考头复现

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py train
.venv-pytorch/bin/python scripts/verify_m3w_hurdle_reference_oof.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_hurdle_reference.py
```

已完成阶段不覆盖。未完成训练加`--resume`，从已保存的8树块继续，不宣称保留
未完成块。私有目录含checkpoint、OOF数组和PID/heartbeat，不能提交Git。
CPU4/interop1/workers0；森林单进程1线程。不能用x86Conda/Rosetta启动训练。

先用原C内部完整来源留出预测挑模型，再全C拟合；所有18任务结束才首次对H预测。
OOF复核要核对216个模型的来源、样本、输入、标签、权重及108组数组，而不只看
脚本成功。H复核核对18组完整输出、共108数组及全部指标。

这里的loss是参考误差MSE和事件Brier，不是轨迹ADE/FDE。预测变好不等于旧策略
安全，不得把概率当逐录像支持保证。旧策略、原2%要求、独立保留数据用途不变。
未启用Stage5C/SMC，没有metric/seconds/true3D/foundation声明。
''')
    print(source.json.dumps(dict(decision=decision,selection_counts=selected,primary=mean_effect),sort_keys=True))


if __name__=='__main__':main()
