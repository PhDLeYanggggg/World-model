"""Render a truthful status report, including explicit not-run dependencies."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_hurdle as run


def put(name, text):
    path = run.PUBLIC/name
    if (run.PUBLIC/'verification.json').exists():
        assert path.read_text() == text, 'A sealed report cannot be rewritten'
    else:
        path.write_text(text)


def value(record):
    if record['point'] is None:
        return 'undefined'
    ci = record['ci95']
    return f"{record['point']:.6g}"+(f" [{ci[0]:.6g}, {ci[1]:.6g}]" if ci else '')


def detailed_readout(s):
    lines=['# Locality, Seed and Tail Readout','',
        'Twelve already-opened development localities. The 3 forecast seeds and repeated source-role views are not independent samples. Intervals use 3,000 nominal paired-locality bootstrap draws, not simultaneous or independent-confirmation intervals. No post-readout selection is performed.','',
        '| Policy | FDE gain/floor % | Worst-view easy gain/CV % | p95 error ratio/floor | Unknown interventions (mean/view) | Entirely abstaining views |',
        '|---|---:|---:|---:|---:|---:|']
    for p in run.POLICIES:
        m=s['summary'][p];w=s['worst_views'][p]
        lines.append(f"| {p} | {value(m['FDE_gain_floor'])} | {w['worst_easy_gain_CV']:.6g} | {value(m['p95_ratio_to_floor'])} | {value(m['unknown_interventions'])} | {w['abstaining_views']} |")
    lines+=['','## Prespecified Primary Contrast','',
        'Positive ADE gain means lower error; positive harm reduction means less positive harm. Fixed-floor-denominator harm is a diagnostic, not a replacement for selected-risk. Intervention counts are matched within each query, not only on average.','',
        '| Locality | Supervised vs marginal matched ADE gain % | Fixed-denominator harm reduction pp | Intervention difference pp |',
        '|---|---:|---:|---:|']
    paired=s['paired']['supervised_matched_vs_marginal_matched']
    for site in paired['ADE_gain_percent']['by_site']:
        vals=[paired[k]['by_site'][site] for k in ('ADE_gain_percent','all_reference_harm_reduction_pp','intervention_difference_pp')]
        lines.append('| '+site+' | '+' | '.join('undefined' if v is None else f'{v:.8g}' for v in vals)+' |')
    lines+=['','## All Seeds Retained','','| Seed | Policy | ADE gain/floor % | Hard gain/floor % |','|---|---|---:|---:|']
    for seed,policies in s['by_seed'].items():
        for p in ('raw_joint','marginal_joint','supervised_joint','raw_matched','marginal_matched','supervised_matched'):
            lines.append(f"| {seed} | {p} | {value(policies[p]['all_gain_floor'])} | {value(policies[p]['hard_gain_floor'])} |")
    lines+=['','## Probability and Conditional Costs','',
        'Quality scores are query-balanced within known-label held-development data. The marginal arm has no direct occurrence label loss; its probability component need not be calibrated. A better Brier score alone is not proof of better joint allocation or a risk certificate.','',
        '| Arm | Easy prevalence | Predicted easy probability | Brier | Log loss | Signed MSE | Signed bias | Conditional reference MSE | Conditional harm MSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in s['quality'].items():
        fields=('easy_rate','predicted_easy_probability','Brier','log_loss','signed_MSE','signed_bias','conditional_reference_MSE','conditional_harm_MSE')
        lines.append('| '+arm+' | '+' | '.join(value(m[k]) for k in fields)+' |')
    lines+=['','No metric, seconds-level, physical-safety, calibration-guarantee or deployment-upgrade claim is inferred. The original incomplete primary remains incomplete; independent roles stay closed.']
    put('locality_seed_quality.md','\n'.join(lines)+'\n')
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='m3w-easy-hurdle-v1'
    matplotlib.rcParams['svg.fonttype']='none'
    names=['supervised_matched_vs_marginal_matched','supervised_matched_vs_raw_matched','marginal_matched_vs_raw_matched']
    labels=['Supervised vs marginal','Supervised vs original','Marginal vs original']
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ax,metric,title in zip(axes,('ADE_gain_percent','all_reference_harm_reduction_pp'),('ADE gain at identical counts (%)','Positive harm reduction (pp)')):
        for i,name in enumerate(names):
            v=s['paired'][name][metric]
            if v['point'] is not None and v['ci95'] is not None:
                point=v['point'];lo,hi=v['ci95']
                ax.plot([lo,hi],[i,i],color=['#197978','#995366','#666666'][i],linewidth=2)
                ax.plot(point,i,'o',color=['#197978','#995366','#666666'][i])
        ax.axvline(0,color='gray',linestyle='--',linewidth=1);ax.set_yticks(range(3),labels);ax.invert_yaxis()
        ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.tick_params(labelsize=9)
    fig.suptitle('Paired easy-risk supervision: frozen common-count controls\n12 development localities; nominal paired bootstrap intervals',fontsize=11)
    buf=io.BytesIO();fig.savefig(buf,format='svg',metadata={'Date':None,'Creator':'M3W deterministic report'})
    dest=run.PUBLIC/'matched_contrasts.svg'
    if (run.PUBLIC/'verification.json').exists():assert dest.read_bytes()==buf.getvalue()
    else:dest.write_bytes(buf.getvalue())
    preview=run.PRIVATE/'figures';preview.mkdir(exist_ok=True)
    fig.savefig(preview/'matched_contrasts.png',dpi=150,metadata={'Software':'M3W deterministic report'});plt.close(fig)


def main():
    cfg, identity = run.identity()
    assert json.loads((run.PUBLIC/'registration.json').read_text()) == identity
    support = json.loads((run.PUBLIC/'fitting_support.json').read_text())
    assert support['identity'] == identity
    rows = support['groups']; rates = [r['easy']/r['known'] for r in rows]
    text = '# Easy Occurrence / Conditional-Risk Experiment\n\n'
    text += ('Registration preceded fitting. Forecasts, floor, utility and all-risk are '
             'cached_verified; fitting-source label audit is fresh_run. The data roles '
             'and 2% risk standard are unchanged.\n\n')
    text += (f"Audited {len(rows)} groups. Easy labels per pair: {min(r['easy'] for r in rows):,}"
             f" to {max(r['easy'] for r in rows):,}. Easy prevalence among known rows: "
             f"{100*min(rates):.3f}% to {100*max(rates):.3f}%. Minimum source easy count: "
             f"{min(n for r in rows for n in r['source_easy_counts'].values())}.\n\n")
    text += (f"Repeated-role row accesses: {sum(r['rows'] for r in rows):,}; known "
             f"{sum(r['known'] for r in rows):,}; unknown excluded from fitting "
             f"{sum(r['unknown'] for r in rows):,}. These are NOT independent examples.\n\n")
    if not (run.PUBLIC/'summary.json').exists():
        completed = (run.PUBLIC/'training_freeze.json').exists()
        checkpoints = len(list((run.PRIVATE/'heads').glob('*/*/checkpoint.pt.gz')))
        execution = json.loads((run.PUBLIC/'execution_status.json').read_text()) if (run.PUBLIC/'execution_status.json').exists() else {}
        status = 'complete_training_freeze_exists' if completed else execution.get('real_training', 'partial' if checkpoints else 'not_run')
        text += (f'## Execution Status\n\nReal neural training: **{status}**; local checkpoints: {checkpoints}. '
                 'Held-development policy readout: **not_run**. No model comparison or accuracy result '
                 'exists for these new heads. Synthetic regression tests are not real training.\n\n'
                 'A fitting-only support audit completed; lack of easy supervision is not '
                 'the blocker. Inspect execution_status.json for the last resource check. '
                 'The isolated CREATE runtime passed an actual optimizer/resume probe; '
                 'its batch-shell startup failure is preserved. Large fitting packets '
                 'are streamed and hash-checked without local temporary files. This '
                 'does not delete old experiments, lower the reserve, reduce scope '
                 'or borrow the simulation project. Neither queued jobs nor completed '
                 'environment probes constitute scientific results.\n\n')
    else:
        s = json.loads((run.PUBLIC/'summary.json').read_text()); assert s['identity'] == identity
        detailed_readout(s)
        text += '## Fresh Development Readout\n\n'
        text += '| Policy | ADE gain/floor (%) | Hard gain/floor (%) | Intervention fraction | Violating / undefined views |\n|---|---:|---:|---:|---:|\n'
        for p in run.POLICIES:
            m, w = s['summary'][p], s['worst_views'][p]
            text += f"| {p} | {value(m['all_gain_floor'])} | {value(m['hard_gain_floor'])} | {value(m['intervention_rate'])} | {w['risk_violating_views']} / {w['undefined_selected_risk_views']} |\n"
        text += '\n## Prespecified Paired Contrasts\n\n'
        for name, m in s['paired'].items():
            text += f"- {name}: ADE gain% {value(m['ADE_gain_percent'])}; fixed-floor harm reduction pp {value(m['all_reference_harm_reduction_pp'])}.\n"
        text += '\n## Easy-Risk Prediction\n\n'
        for arm, m in s['quality'].items():
            text += f"- {arm}: Brier {value(m['Brier'])}; log loss {value(m['log_loss'])}; signed MSE {value(m['signed_MSE'])}.\n"
        text += f"\nExploratory screen pass: {s['gates']['exploratory_screen_pass']}. No deployment change.\n\n"
    text += ('## Boundaries\n\nIndependent selection/calibration/confirmation remain closed. '
             'The original incomplete primary is not replaced. Proper probability loss '
             'does not prove calibration, risk coverage or cross-domain generalization. '
             'No threshold search. Coordinates remain image-local detector silver; '
             'obs8/pred12 uses raw-frame stride12. No metric/seconds, human-gold, physical '
             'safety, true3D, foundation or submission-readiness claim. Stage5C/SMC disabled.\n')
    put('results.md', text)
    put('operation_zh.md', '''# 复现与恢复说明

本轮是两种 easy 风险监督目标的配对实验，不重训预测器。既有产物按 hash 复用；
训练源审计是本轮重新计算。未生成 summary 时，不得把训练完成写成开发评价完成。

使用仓库根目录的 arm64 `.venv-pytorch/bin/python`。CPU4、interop1、workers0。
保留10GiB空闲，入口会拒绝越过这一底线；不要更改配置绕过。CREATE 需先确认独立
M3W目录、环境及允许的调度器资源，不借用 simulation 的目录或登录节点训练。

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_hurdle.py --phase pilot
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_hurdle.py --phase train --resume
```

训练前已完成登记与 fitting-only audit。pilot100步若已留下checkpoint，不重复无resume
启动；恢复时原命令加`--resume`。完整训练216头，每头2,000更新，不静默缩减。每500步
原子保存模型、优化器、随机状态和采样摘要，记录PID和心跳。中断保留已完成组。

全部训练完成后，先将training_freeze.json安全提交，再运行`--phase decide --resume`。
全部动作完成后，先提交decision_freeze.json，再运行`--phase evaluate`。不得交换顺序。
本轮训练在 CREATE 完成，因此完整训练重放也在相同 CREATE runtime 完成；不要把
本机不同架构训练当作逐位复现。其他本机训练实验可用`--phase replay_fit`重训首对。
本轮本机只执行`--phase replay_decide`和`--phase replay_evaluate`。
训练重放只对第一组两头完整重训；其余组是检查点核对及推理重放，不写成全216头重训。

独立算术检查使用scripts/verify_m3w_easy_hurdle.py。冻结前生成报告使用
scripts/report_m3w_easy_hurdle.py。检查点、行级数据和原始调度器输出不进Git。
旧完整测试集有写报告/真实训练副作用；先运行本轮scoped tests，不宣称全仓库通过。

## 当前 CREATE 路径

本地 pilot 因10GiB保留空间不足，在第一次真实更新前退出。不得删旧结果或降保留线。
独立 M3W CREATE CPU环境已通过真实优化器和精确恢复合成检查；不借 simulation 环境。
训练数据按组从内存上传，重启时逐一检查已有远端hash。连接中断不是训练失败。

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.export_m3w_easy_hurdle_create --resume
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.submit_m3w_easy_hurdle_create --phase pilot
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.inspect_m3w_easy_hurdle_create --phase pilot
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.submit_m3w_easy_hurdle_create --phase train
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.inspect_m3w_easy_hurdle_create --phase train
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.collect_m3w_easy_hurdle_training --phase train
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.submit_m3w_easy_hurdle_verification
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.collect_m3w_easy_hurdle_training --phase verify
```

这些是分阶段操作，不是可盲目重复的命令串。只有108组传输完成才提交pilot；只有真实
pilot完成且调度器exit0才提交train。已存在submission receipt或不确定提交时只检查，
不重复提交。训练使用已登记的同一fit实现、CPU4/interop1/workers0和原432,000更新预算。
远端训练完成不等于开发评价完成。模型与动作必须分别hash冻结并提交后才读取对应结果。
本地磁盘不足期间，远端模型和行级数组不批量回传。独立确认仍关闭。

本轮pilot前两次SSH在握手时失败，保留原receipt。远端只读确认无intent、receipt、脚本
或排队作业后，第3次实际提交成功；查看它需`--phase pilot --attempt 3`，不重新提交。
完整训练只提交一次。`create_training_freeze.json`记录远端216个checkpoint的hash；
核验job重新核对全部头，第一组两头从头重训4,000更新，并逐字段对比，耗时字段除外。
这不是全216头重复训练，也不是held预测重放；后两者的状态不得混写。

本地空间恢复后，使用`scripts/restore_m3w_easy_hurdle_heads.py`仅取回216个已冻结的
检查点，共89,517,817bytes。恢复程序逐文件核对远端已提交清单的hash，不传输4.91GB
训练包，不覆盖不同产物，不更改10GiB保留线。恢复依赖远端训练/核验记录已提交。
本地training_freeze提交后才运行原始decide；decision_freeze提交后才运行evaluate。
fit_replay.json明确引用CREATE首对4,000更新逐字段重放，不称跨架构重训等价。
prediction_replay和evaluation_replay才是本轮本地重新运行的动作及评价重放。
''')


if __name__ == '__main__':
    main()
