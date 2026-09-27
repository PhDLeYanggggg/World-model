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


def main():
    cfg, identity = run.identity()
    assert json.loads((run.PUBLIC/'registration.json').read_text()) == identity
    support = json.loads((run.PUBLIC/'fitting_support.json').read_text())
    assert support['identity'] == identity
    rows = support['groups']; rates = [r['easy']/r['known'] for r in rows]
    text = '# Easy Occurrence / Conditional-Risk Experiment\n\n'
    text += ('Registration preceded fitting. Forecasts, floor, utility and all-risk are '
             'cached_verified; fitting-source label audit is fresh_run. The data roles '
             'and2%risk standard are unchanged.\n\n')
    text += (f"Audited{len(rows)}groups. Easy labels per pair: {min(r['easy'] for r in rows):,}"
             f" to {max(r['easy'] for r in rows):,}. Easy prevalence among known rows: "
             f"{100*min(rates):.3f}% to {100*max(rates):.3f}%. Minimum source easy count: "
             f"{min(n for r in rows for n in r['source_easy_counts'].values())}.\n\n")
    text += (f"Repeated-role row accesses: {sum(r['rows'] for r in rows):,}; known "
             f"{sum(r['known'] for r in rows):,}; unknown excluded from fitting "
             f"{sum(r['unknown'] for r in rows):,}. These are NOT independent examples.\n\n")
    if not (run.PUBLIC/'summary.json').exists():
        completed = (run.PUBLIC/'training_freeze.json').exists()
        checkpoints = len(list((run.PRIVATE/'heads').glob('*/*/checkpoint.pt.gz')))
        status = 'complete_training_freeze_exists' if completed else 'partial' if checkpoints else 'not_run'
        text += (f'## Execution Status\n\nReal neural training: **{status}**; checkpoints: {checkpoints}. '
                 'Held-development policy readout: **not_run**. No model comparison or accuracy result '
                 'exists for these new heads. Synthetic regression tests are not real training.\n\n'
                 'A fitting-only support audit completed; lack of easy supervision is not '
                 'the blocker. Inspect execution_status.json for the last resource check. '
                 'Storage and a permitted compute placement must be resolved '
                 'without deleting old experiments, lowering the reserve, reducing scope '
                 'or using unapproved remote paths. Completed sources and reports remain preserved.\n\n')
    else:
        s = json.loads((run.PUBLIC/'summary.json').read_text()); assert s['identity'] == identity
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
训练源审计是本轮重新计算。未生成 summary 时，真实训练和开发评价都不得写成完成。

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
然后执行`--phase replay_fit`、`--phase replay_decide`、`--phase replay_evaluate`。
replay_fit只对第一组两头完整重训；其余组是推理重放，不写成全216头重训复现。

独立算术检查使用scripts/verify_m3w_easy_hurdle.py。冻结前生成报告使用
scripts/report_m3w_easy_hurdle.py。检查点、行级数据和原始调度器输出不进Git。
旧完整测试集有写报告/真实训练副作用；先运行本轮scoped tests，不宣称全仓库通过。
''')


if __name__ == '__main__':
    main()
