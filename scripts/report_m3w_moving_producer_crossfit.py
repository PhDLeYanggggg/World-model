"""Summarize completed donor construction without promoting development results."""
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_producer_crossfit as run


def validate(analysis, fits, replay):
    if (analysis['models'] != 6 or len(fits) != 6 or analysis['optimizer_updates'] != 24000
            or not analysis['donor_bank_complete'] or not replay['six_predictions_exact']
            or analysis['downstream_risk_lift'] != 'not_run' or analysis['deployment_changed']):
        raise ValueError('Do not report incomplete work or a risk success')
    for f in fits:
        if f['result']['step'] != 4000 or not f['result']['complete'] or f['held_sample_draws']:
            raise ValueError('Incomplete fit or teacher leakage')


def figures(analysis, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for f in fits:
        loss = f['result']['losses']
        axes[0].plot([r['step'] for r in loss], [r['loss'] for r in loss],
                     linewidth=1, label=f['job']['key'].replace('eqmotion_', '').replace('_seed17', ''))
    axes[0].set(xlabel='Optimizer updates', ylabel='Normalized fitting batch loss',
                yscale='log', title='Six fixed-budget trajectory fits')
    axes[0].legend(fontsize=8)
    sources = [s for f in analysis['forecasts'] for s in f['sources']]
    ratios = [s['neural_ADE']/s['causal_CV_ADE'] for s in sources]
    axes[1].bar(range(len(sources)), ratios, color=['#2563eb']*6+['#15803d']*8)
    axes[1].axhline(1, color='#dc2626', linestyle='--', linewidth=1.2, label='Causal CV')
    axes[1].set(xticks=range(len(sources)), xticklabels=[s['source'].split('_')[-1] for s in sources],
                xlabel='Donor source component', ylabel='OOF neural ADE / causal CV ADE',
                title='Training donors, not outer confirmation')
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    folder = run.PRIVATE/'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/'training_and_donors.png', dpi=170)
    fig.savefig(folder/'training_and_donors.pdf')
    plt.close(fig)


def main():
    p = run.PUBLIC
    reg, analysis, population, pilot, replay = [run.read(p/name) for name in
        ('registration.json', 'analysis.json', 'population.json', 'pilot.json', 'replay.json')]
    fits = [run.read(p/'fits'/f'{j["key"]}.json') for j in reg['jobs']]
    validate(analysis, fits, replay)
    for f in fits:
        run.checked(f['checkpoint'])
    for forecast in analysis['forecasts']:
        run.checked(forecast['forecast'])
    sources = [s for f in analysis['forecasts'] for s in f['sources']]
    worse = [s['source'] for s in sources if s['neural_ADE'] > s['causal_CV_ADE']]
    better = sum(s['neural_ADE'] < s['causal_CV_ADE'] for s in sources)
    table = ['| Model | Fit sources | Donor sources | Donor rows | First / final batch loss | Fit seconds |',
             '|---|---:|---:|---:|---:|---:|']
    trace = io.StringIO()
    writer = csv.writer(trace)
    writer.writerow(['model', 'step', 'fit_batch_loss', 'gradient_norm', 'learning_rate'])
    total_seconds = 0.
    for f, forecast in zip(fits, analysis['forecasts']):
        job, result = f['job'], f['result']
        loss = result['losses']
        total_seconds += result['seconds']
        table.append(f'| {job["key"]} | {len(job["fit_sites"])} | {len(job["donor_sites"])} | '
            f'{forecast["rows"]} | {loss[0]["loss"]:.6f} / {loss[-1]["loss"]:.6f} | {result["seconds"]:.1f} |')
        for row in loss:
            writer.writerow([job['key'], row['step'], row['loss'], row['gradient_norm'], row['learning_rate']])
    report = f'''# Source-Held Forecast Donors Inside Producer Training

I trained six new EqMotion controls to expand risk-learning source coverage
without moving outer calibration or evaluation sources into training.
Each model completed 4,000 updates; primary total 24,000, with a further 40
updates for the direct recovery control. One seed, not a three-seed result.

**The donor bank is complete. Better safe intervention is not yet established.**

The bank contains {analysis['rows']:,} rows from 14 source components, partitioned
by parent: 6 sources for producer A0 and 8 for producer A1, never blindly pooled
across parent contexts. {population['known_rows']:,} rows have some future labels,
{population['complete_rows']:,} have all 12, and {population['unknown_rows']:,}
have no valid future labels. Unknown is not safe. Row IDs, teacher ancestry and
input/label arrays are recorded separately. Inputs remain past-only by annotation
index; the source annotation-generation process is not proved temporally causal.

## Training

{chr(10).join(table)}

Summed primary fitting time including the reused pilot prefix: {total_seconds:.1f}s.
First/final values are sampled batch losses, not an epoch-level comparison or
proof of convergence. Every final checkpoint is fixed at 4,000 updates.
All nonfitting source sample draws are zero. Six forecast caches reproduce
exactly from final checkpoints. Replay is verification, not independent evidence.

## Why Not Pool The Old Caches?

The metadata-only compatibility matrix rejects direct cross-context pooling.
A shared teacher does not make another context's outer H a training donor.
Different teachers can have seen the intended critic sources and can use
different training-defined risk targets. The new forecasts instead come from
models fitted only inside the appropriate parent A, excluding each donor source.

These are smaller source-trained teachers: each fits four to six sources,
versus the original six or eight. Their error distribution can differ.
More sources do not automatically yield calibrated risk estimates.
OOF donor ADE is lower than causal CV on {better}/14 sources and higher on
{len(worse)}/14: {', '.join(worse)}. These negative sources are retained, not
filtered after observing their errors. This is not a strongest-baseline result.

## Evidence Status

- fresh_run: subset packing, six real Torch fits, OOF predictions and donor diagnostics.
- cached_verified: pinned source admission, partitions, code and source hashes.
- not_run: augmented risk-head comparison, new outer C/H readout and deployment.
- The original 2% all/easy harm, signed easy-net and support requirements remain.
- Empty controls remain NA. No identity-fallback certificate. Old failures remain.
- Original 24-group main estimand remains NA; this does not repair group068.
- Native obs8/pred12 image-coordinate silver labels; no meters, verified seconds,
  true3D, foundation, Stage5C execution, SMC or submission-ready claim.

## Next Experiment

Compare the same risk model and fixed decision rule with own-B-only donors
versus parent-A OOF augmentation, on the already-exposed own-B development
screen. Preserve query-source exclusions and compare teacher-size effects.
Do not open a new test or select thresholds on outcomes. A failed donor bank
comparison would reject the coverage remedy; a successful one still needs
independent confirmation under the original complete policy contract.

[Protocol](protocol.md) | [Source diagnostics](donor_diagnostics.md)
| [Loss trace](loss_trace.csv) | [Reproduction](operation_zh.md)
'''
    (p/'README.md').write_text(report)
    (p/'loss_trace.csv').write_text(trace.getvalue())
    audit = ['# Metadata-Only Cache Eligibility', '',
             'No new forecast or future-label readout is used by this matrix.', '',
             '| Target | Donor context | Blind pooling allowed | Reasons |', '|---|---|---|---|']
    for row in reg['eligibility']:
        audit.append(f'| {row["target"]} | {row["donor"]} | {row["blind_pooling_allowed"]} | {", ".join(row["reasons"])} |')
    (p/'eligibility_report.md').write_text('\n'.join(audit)+'\n')
    diagnostic = ['# OOF Training-Donor Diagnostics', '',
        'Not a new outer evaluation, not a deployment comparison. ADE uses available',
        'future labels; FDE requires the endpoint. Unknown rows are counted separately.',
        'All units image coordinates, native8/12 steps, not verified seconds.', '',
        '| Source | Rows | Known | Unknown | Neural ADE | CV ADE | Neural FDE | CV FDE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in analysis['forecasts']:
        for s in f['sources']:
            values = [s[k] for k in ('source', 'rows', 'known_rows', 'unknown_rows',
                                    'neural_ADE', 'causal_CV_ADE', 'neural_FDE', 'causal_CV_FDE')]
            diagnostic.append('| '+' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)+' |')
    (p/'donor_diagnostics.md').write_text('\n'.join(diagnostic)+'\n')
    figures(analysis, fits)
    (p/'paper_addendum.md').write_text('''# Within-Producer Crossfitting: Data Repair, Not A New Main Result

The preceding fixed-budget continuation improved fitting loss in eight of eight
risk heads, but improved source-held plain MSE in only two and failed all three
decision screens. The fitting banks had only one or two source components.
This motivates a source-coverage hypothesis, not a demonstrated causal explanation.

We construct additional forecast-risk training donors strictly inside each
original producer-training set A. For each of three predetermined source
partitions, a randomly initialized EqMotion control is fitted on the other
partitions and forecasts only the excluded sources. Baseline selection and loss
normalization use the fitting partitions. Two parent sets require six models,
one seed, 4,000 updates each. The resulting 125,540-row bank contains 6 and 8 source
components respectively. Teacher ancestry remains separate across parent contexts.
No outer C/H source is used by its parent's producer, and no new outer readout
is made. Inputs contain observed history and current-visible neighbors only;
future labels and validity masks are supervision, not model inputs.

Every primary model completes its budget. Sampling logs show zero draws from
excluded sources; saved forecasts replay exactly. The first recovery check
failed only because an intermediate stop adds a log row. All model, optimizer,
RNG and sampling states and common loss records agree exactly; the extra log
and initial failure are preserved. This is an engineering verification, not an
independent experiment. The donor errors versus causal CV are reported for all
sources, including negative cases and unknown outcomes.

Risk-model improvement from these donors remains untested. Smaller source-trained
teachers may create a new risk-distribution mismatch. A fixed downstream
augmentation control on already-exposed development records is required before
any independent confirmation. The original 2% full-policy risk and support rules,
empty-support NA and original 24-group primary NA remain unchanged. No identity
exemption or deployment update is justified. Native image-coordinate 8/12 silver
labels do not establish metric dynamics, verified elapsed seconds or a
foundation world model. This addendum does not change the compiled main paper.
''')
    (p/'operation_zh.md').write_text('''# Producer-A 交叉拟合复现

使用仓库内原生 arm64 `.venv-pytorch/bin/python`，CPU4、inter-op1、workers0。
所有命令在项目根目录执行。旧文件与数据哈希须一致，不能用新数据覆盖原实验。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py pilot
.venv-pytorch/bin/python scripts/verify_m3w_producer_crossfit_pilot.py
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py forecast
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_producer_crossfit.py
```

上述是空实验目录的历史执行顺序，不要在已经完成的目录重新运行 pilot。
原 pilot 检查器会因恢复路线多出第20步日志退出1；此失败记录保留。
随后的专用验证器核实所有训练状态、共同日志和推理输出逐项完全一致；
没有删除日志，也没有重新训练来覆盖差异。完整模型只使用原40步恢复前缀。
现有结果只需运行 replay 和 reporter，不必重复训练。

中断后检查 `data/stage_cvpr2027_experiments/moving_producer_crossfit_v1/heartbeat.json`
与 PID 是否仍运行。不要重复启动健康任务。使用相同代码、配置和数据加
`train --resume` 从最近原子检查点恢复；不选取中途效果最好的 checkpoint。
每200步保存、每50步写心跳，健康进程允许12小时。磁盘保留5GiB。

缓存、原始标注、checkpoint 留在本地，不上传Git。新缓存仅在原预测器A内部
交叉拟合，不能据此把外层C/H或其他受保护来源移入训练。原2%规则、空支持NA、
旧失败、原24组主估计NA全部保留。训练完成不等于风险头提升或投稿准备完成。
''')
    run.once(p/'report_receipt.json', dict(analysis=run.artifact(p/'analysis.json'),
        reporter=run.artifact(Path(__file__)), reports=[run.artifact(p/name) for name in
        ('README.md', 'loss_trace.csv', 'eligibility_report.md', 'donor_diagnostics.md', 'operation_zh.md', 'paper_addendum.md')]))
    print('Completed donor construction report; downstream risk lift remains not_run.')


if __name__ == '__main__':
    main()
