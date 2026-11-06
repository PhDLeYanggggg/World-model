"""Report complete matched actor evidence from public frozen scalar receipts."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_visual_actor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_visual_actor_v1'
SOURCE = ROOT/'outputs/publication_readiness_2026_09/moving_producer_crossfit_v1'


def read(path): return json.loads(path.read_text())
def num(value): return 'NA' if value is None else f'{value:+.4f}'
def macro(values): return None if any(v is None for v in values) else statistics.mean(values)


def plots(a, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for arm, color in (('raw', '#525252'), ('quality_geometry', '#7c3aed'), ('motion_geometry', '#0f766e')):
        traces = []
        for f in fits:
            if arm == 'raw' and f['arm'] == 'motion_geometry':
                traces.append(read(SOURCE/'fits'/f'{f["job"]["key"]}.json')['result']['losses'])
            elif f['arm'] == arm: traces.append(f['result']['losses'])
        common = sorted(set.intersection(*[set(x['step'] for x in t) for t in traces]))
        by_step = [{x['step']: x['loss'] for x in t} for t in traces]
        axes[0].plot(common, [statistics.median(t[s] for t in by_step) for s in common], color=color, label=arm)
    axes[0].set(title='Median across six fitting designs, one seed', xlabel='Optimizer updates', ylabel='Training objective, not validation')
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    rows = sorted([r for r in a['sources'] if r['contrast'] == 'primary'], key=lambda r: r['source'])
    known = [r['summary'][-1]['paired_known_pp'] for r in rows]
    lower = [r['summary'][-1]['paired_completion_lower_pp'] for r in rows]
    ix = list(range(len(rows)))
    axes[1].bar([x-.2 for x in ix], known, width=.4, color='#2563eb', label='Known difference')
    axes[1].bar([x+.2 for x in ix], lower, width=.4, color='#b45309', label='Paired completion lower')
    axes[1].set_xticks(ix, [r['source'].rsplit('_', 1)[-1] for r in rows], rotation=60)
    axes[1].axhline(0, color='#525252', linewidth=.6)
    axes[1].set(title='Motion minus quality: all14source donors', xlabel='Source component', ylabel='Whole available12 ADE difference (pp)')
    axes[1].grid(axis='y', alpha=.2); axes[1].legend(fontsize=8)
    folder = PRIVATE/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'actor_training_and_sources.png', dpi=170); plt.close(fig)


def main():
    a = read(PUBLIC/'analysis.json'); p = a['primary']
    fits = [read(path) for path in sorted((PUBLIC/'fits').glob('*.json'))]
    verify = read(PUBLIC/'verification.json'); control = read(PUBLIC/'raw_control_refit.json')
    if len(fits) != 12 or not all(f['result']['complete'] and f['result']['step'] == 4000 for f in fits) or verify['models'] != 2 or not verify['full_refits_exact']:
        raise ValueError('Complete registered full-budget actors and both actual full refits required')
    screen = p['component_progression_screen_pass']; interval = p['interval95']
    resource_receipt = PUBLIC/'resource_amendment.json'
    repeated_updates = read(resource_receipt)['extra_discarded_updates'] if resource_receipt.exists() else 0
    lines = ['# Does Past-Image Context Improve The Trajectory Actor?', '',
        'I moved the same verified past-image measurements into the trajectory',
        'predictor, rather than only the intervention controller. This experiment',
        'changes the actual neural actor and retains the original whole12loss.', '',
        f'Component progression screen: **{"PASS, exploratory" if screen else "FAIL"}**.',
        f'Primary motion-minus-quality known difference: **{num(p["known_pp"])}pp**.',
        f'Paired completion lower: **{num(p["completion_lower_pp"])}pp**, exploratory95%',
        f'source-bootstrap interval **[{num(interval[0])}, {num(interval[1])}]**.',
        f'Known-positive sources:{p["positive_known_sources"]}/14. Fold means:{p["fold_known_means"]}.', '',
        '**This is not a deployment result or independent confirmation.** No new',
        'controller is trained. The original2%/support/NA requirements remain.', '',
        '## Full Paired Prefix Matrix', '',
        '| Contrast | Prefix | Control gain vs CV % | Treatment gain vs CV % | Paired known pp | Paired lower pp |',
        '|---|---:|---:|---:|---:|---:|']
    for contrast in ('primary', 'quality_bundle', 'motion_bundle'):
        rr = [r for r in a['sources'] if r['contrast'] == contrast]
        for i, prefix in enumerate((1, 3, 6, 12)):
            m = [r['summary'][i] for r in rr]
            values = [macro([x['control']['known_gain_percent'] for x in m]), macro([x['treatment']['known_gain_percent'] for x in m]),
                macro([x['paired_known_pp'] for x in m]), macro([x['paired_completion_lower_pp'] for x in m])]
            lines.append(f'| {contrast} | {prefix} | '+' | '.join(num(v) for v in values)+' |')
    lines += ['', 'Primary is full motion versus the matched quality/identity-motion control.',
        'The two raw comparisons are bundle contrasts, not pure motion attribution.',
        'Weights balance queries within sources, then average sources. The lower',
        'statistic bounds missing-label uncertainty; it is not the actual unknown',
        'forecast error. Prefix-only improvements cannot replace a failed whole12.', '',
        '## Every Source', '',
        '| Source | Contrast | Rows | Known | Partial | Unknown | Control gain % | New gain % | Paired pp | Paired lower pp | New easy +harm % | New signed easy loss % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['sources']:
        s = r['summary'][-1]
        values = [s['control']['known_gain_percent'], s['treatment']['known_gain_percent'], s['paired_known_pp'], s['paired_completion_lower_pp'],
            s['treatment']['known_easy_positive_harm_percent'], s['treatment']['known_easy_signed_degradation_percent']]
        lines.append(f'| {r["source"]} | {r["contrast"]} | {s["rows"]} | {s["known_rows"]} | {s["partial_rows"]} | {s["wholly_unknown_rows"]} | '+
            ' | '.join(num(v) for v in values)+' |')
    raw_harm = [r['summary'][-1]['treatment']['known_easy_positive_harm_percent'] for r in a['recordings'] if r['contrast'] == 'primary']
    lines += ['', f'Raw motion actor known-easy +harm exceeds2%in{sum(x is not None and x > 2 for x in raw_harm)}/34recordings.',
        'This is a raw actor diagnostic, not the full selected-policy certificate.',
        'Undefined references remain NA. Positive-harm ratios can exceed100%;',
        'they are not probabilities,physical safety or an identity certification.', '',
        '## What Actually Ran', '',
        'Fresh_run:twelve full4000update image-conditioned actors,one full4000update',
        'raw-control refit and two full4000update treatment refits on the first',
        'fixed source design. Do not call these twelve complete exact refits.',
        'The real100+100versus200recovery pilot is exact;its200prefix is resumed',
        'into its own full motion fit. A separate direct200pilot is additional.',
        f'Total actual optimizer updates:{60200+repeated_updates:,}including recovery,full refits and discarded/repeated updates.',
        f'Disk-guard recovery repeats{repeated_updates}updates from the last atomic checkpoint;the scientific endpoint stays4000.',
        'Each treatment has3,028,996nominal/582,404trainable parameters;unused',
        'author heads remain frozen. The image branch adds1,728parameters.',
        f'Raw shared core/optimizer/sampler/RNG/common trace exact:{control["shared_core_optimizer_sampler_RNG_and_common_trace_exact"]}.', '',
        'Cached_verified:six original raw OOF predictors and their125540donor',
        'forecasts,the source split and past-image measurements. Fresh preparation',
        'joins46causal action-invariant features to every original global ID.',
        'Normalizers use past fitting rows only;no donor or own-B statistics.', '',
        'Not_run:new risk head fitting,new own-B/C/H/reserved readout,three forecaster',
        'seeds,independent confirmation or a complete new deployment risk check.',
        'All models use native arm64 CPU4/inter-op1/workers0. Final4000checkpoint,',
        'no early stopping,threshold tuning,quick downgrade or cohort deletion.', '',
        '## Scope', '',
        'Native MovingDroneCrowd++ observation8/prediction12,stride1,image',
        'coordinates and available-label ADE. Not SDDt50,meters,seconds,true3D,',
        'foundation,human-gold or physical safety. The adapted image-conditioned',
        'model does not automatically inherit the original core equivariance proof.',
        'Bounding a latent offset does not bound trajectory error. The branch is',
        'conventional conditioning,not a novel world-model method by itself.',
        'The current paper is not submission-ready. Stage5C execution/SMC remain off.', '',
        '[Protocol](protocol.md) | [Model scope](model_scope.md) | [Loss trace](loss_trace.csv) |',
        '[Recording diagnostics](recording_diagnostics.csv) | [Operation](operation_zh.md)']
    (PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    fields = ('job', 'arm', 'step', 'loss', 'gradient_norm', 'learning_rate', 'supported_batch_rows', 'mean_past_normalized_ADE')
    with (PUBLIC/'loss_trace.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for f in fits:
            for row in f['result']['losses']: writer.writerow(dict(job=f['job']['key'], arm=f['arm'], **row))
    records = [dict(job=r['job'], producer_fold=r['producer_fold'], contrast=r['contrast'], recording=r['recording'], **s,
        control_gain_percent=s['control']['known_gain_percent'], new_gain_percent=s['treatment']['known_gain_percent'])
        for r in a['recordings'] for s in r['summary']]
    with (PUBLIC/'recording_diagnostics.csv').open('w', newline='') as out:
        scalar_fields = [k for k in records[0] if k not in ('control', 'treatment')]
        writer = csv.DictWriter(out, fieldnames=scalar_fields); writer.writeheader()
        writer.writerows({k: r[k] for k in scalar_fields} for r in records)
    gates = ['# Past-Image Actor Component Gates', '', '| Gate | Status | Evidence |', '|---|---|---|',
        '| Actual real Torch fitting | PASS | Twelve actors at4000updates,not NumPy fallback |',
        '| Source exclusion and causal inputs | PASS within pipeline | No held draws,train-only normalizers,strict input schema |',
        '| Full old-control parity | PASS | Shared scientific state and common trace exact |',
        '| Real interrupted recovery | PASS |100+100equals200 |',
        '| Two full treatment refits | PASS | Fixed first design,not all twelve |',
        '| Whole12motion component | '+('PASS, exploratory' if screen else 'FAIL')+' | Registered paired lower/source/fold screen |',
        '| New full policy risk | NOT RUN | No new controller or own-Breadout |',
        '| Independent confirmation/three seeds | NOT RUN | Existing A donors,one forecaster seed |',
        '| Deployment/submission readiness | FALSE | Original requirements remain unmet |',
        '| Stage5C execution/SMC | FALSE | Neither executed |']
    (PUBLIC/'world_model_gate_visual_actor.md').write_text('\n'.join(gates)+'\n')
    failure = ['# Actor Conditioning Result And Remaining Failure', '',
        f'The registered upstream component screen is{screen}.',
        f'Whole12known difference:{num(p["known_pp"])}pp;conservative lower:{num(p["completion_lower_pp"])}pp.',
        f'Exploratory interval:[{num(interval[0])},{num(interval[1])}].', '',
        'The architecture changes an actual trajectory predictor,not only a cost',
        'head or threshold. Matched quality/motion models share initialization,',
        'draws,budget and original loss. A positive point without its source/fold',
        'and completion checks would not support progression. A failed component',
        'does not prove that images contain no useful information.', '',
        'Model-side candidate mechanisms include weak background proxy,limited',
        'source support,poor placement of the context branch,undertraining or a',
        'mismatch between average forecasting and easy-case harm. These are',
        'hypotheses unless separately tested;do not assert one as a proven cause.',
        'This experiment does not isolate raster semantics or train pixels.', '',
        'The original controller still needs reliable source-held harm/reference',
        'estimation and independent risk calibration. Whole12component lift would',
        'not prove its2%recording contract. Raw actor harm is retained separately.',
        'No outcome-selected threshold,new test access or deployment change.',
        'Previous prefix-balanced and visual-cost failures remain in the ledger.']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure)+'\n')
    operation = ['# 本轮真实轨迹训练复现', '',
        '使用现有 arm64 `.venv-pytorch`，CPU4/inter-op1/workers0。',
        '需要本地已校验的来源轨迹、过去图像派生特征及 pinned EqMotion 代码。',
        '数组、缓存、第三方源码和 checkpoint 不在 Git 中。', '',
        '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_visual_actor.py control --resume',
        '.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py train --resume',
        '.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py verify --resume',
        '.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py forecast',
        '.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py evaluate',
        '.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py replay',
        '.venv-pytorch/bin/python scripts/report_m3w_moving_visual_actor.py',
        '```', '',
        '已完成的 pilot/evaluate 收据不可覆盖重跑；核验已有产物即可。',
        '训练在同定义断点上恢复，已完成模型校验后复用；不得以 quick 替代4000步。',
        'resource_amendment 只将磁盘工程预留由5GiB改为2GiB，保留1GiB产物上限。',
        '原模型、训练终点和2%科学评价规则不变，不删除旧数据。',
        'replay 重算所有预测和原指标，不进行新优化。',
        'loss CSV 的 mean_past_normalized_ADE 是旧字段名，实际为未加权 batch',
        'available-label ADE 的原始 packed 图像坐标值，不是米或不同场景统一尺度。',
        '当前是训练侧来源排除组件实验，不是新的独立测试或部署结果。']
    (PUBLIC/'operation_zh.md').write_text('\n'.join(operation)+'\n')
    plots(a, fits)
    print('All twelve actor fits and every source/recording reported; no deployment claim.')


if __name__ == '__main__': main()
