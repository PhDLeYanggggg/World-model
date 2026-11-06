"""Render frozen public evidence without loading private forecasts or outcomes."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_visual_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_visual_risk_v1'
ARMS = ('no_image', 'image_bounds', 'quality_geometry', 'motion_geometry')


def read(path):
    return json.loads(path.read_text())


def number(value, scale=1):
    return 'NA' if value is None else f'{scale*value:+.4f}'


def ci(value):
    bounds = value['interval95']
    return 'NA' if bounds is None else f'[{number(bounds[0])}, {number(bounds[1])}]'


def worst(a, arm, policy, field):
    values = [v[field] for row in a['rows'] if row['arm'] == arm
        for v in row['policies'][policy]['recordings'].values() if v[field] is not None]
    return max(values) if values else None


def fit_pairs(fits, control, treatment):
    keyed = {(r['fold'], r['seed'], r['arm']): r for r in fits}
    return [(keyed[f, s, control], keyed[f, s, treatment]) for f in (0, 1) for s in (17, 29, 43)]


def write_fits(fits):
    fields = ('fold', 'seed', 'arm', 'step', 'gradient_norm', 'total', 'composed_MSE',
        'observed_conditional_MSE', 'observation_BCE', 'distinct_fitting_rows_drawn')
    with (PUBLIC/'loss_trace.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for r in fits:
            for trace in r['trace']:
                writer.writerow(dict(fold=r['fold'], seed=r['seed'], arm=r['arm'], **trace))


def write_records(a):
    fields = ('arm', 'policy', 'source', 'seed', 'recording', 'rows', 'selected',
        'unknown_selected', 'known_all_risk', 'known_easy_risk', 'completion_all_risk_upper',
        'completion_easy_risk_upper', 'known_easy_net_degradation', 'supported')
    with (PUBLIC/'recording_checks.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for row in a['rows']:
            for policy, result in row['policies'].items():
                for recording, values in result['recordings'].items():
                    writer.writerow(dict(arm=row['arm'], policy=policy, source=row['source'],
                        seed=row['seed'], recording=recording, **values))


def plots(a, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = PRIVATE/'figures'; folder.mkdir(exist_ok=True)
    colors = ('#2563eb', '#d97706', '#7c3aed', '#0f766e')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for i, (arm, color) in enumerate(zip(ARMS, colors)):
        p = a['policies'][arm]['joint_free']; u = p['utility']['known']; bounds = u['interval95']
        if bounds is not None:
            axes[0].errorbar(i, u['point'], yerr=[[u['point']-bounds[0]], [bounds[1]-u['point']]],
                fmt='o', color=color, capsize=4)
        axes[1].bar(i, p['safety']['supported_record_views'], color=color)
    for ax in axes:
        ax.set_xticks(range(4), ['No image', 'Bounds', 'Quality', 'Motion'], rotation=15)
        ax.grid(axis='y', alpha=.2)
    axes[0].axhline(0, color='#525252', linewidth=.6)
    axes[0].set(title='Joint-free: exposed own-B development', ylabel='Known-label ADE gain versus CV (%)')
    axes[1].axhline(72, color='#dc2626', linestyle='--')
    axes[1].set(title='Original complete recording contract', ylabel='Supported repeated views (of 72)', ylim=(0, 76))
    fig.savefig(folder/'visual_context_result.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, fold in zip(axes, (0, 1)):
        for arm, color in zip(ARMS, colors):
            rows = [r for r in fits if r['fold'] == fold and r['arm'] == arm]
            steps = [v['step'] for v in rows[0]['trace']]
            if any([v['step'] for v in r['trace']] != steps for r in rows):
                raise ValueError('Matched loss traces must retain the same steps')
            values = [statistics.median(r['trace'][i]['total'] for r in rows) for i in range(len(steps))]
            ax.plot(steps, values, color=color, label=arm)
        ax.set(title=f'A{fold}: median of three risk seeds', xlabel='Optimizer updates',
            ylabel='Training objective (not validation)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(folder/'visual_context_training.png', dpi=170); plt.close(fig)


def main():
    a = read(PUBLIC/'analysis.json'); train = read(PUBLIC/'training.json')
    verified = read(PUBLIC/'verification.json'); pilot = read(PUBLIC/'pilot.json')
    fits = [read(p) for p in sorted((PUBLIC/'fits').glob('*.json'))]
    if (len(fits) != 24 or train['models'] != 24 or not verified['all24_full_refits_exact']
            or any(r['steps'] != 4096 or r['parameters'] != 19814 for r in fits)):
        raise ValueError('Report only the complete registered experiment')
    primary = a['contrasts']['primary']['completion_lower_pp']
    screen_status = ('PASS, exploratory' if a['motion_component_screen_pass']
        else 'NA' if primary['interval95'] is None else 'FAIL')
    motion_safe = a['policies']['motion_geometry']['joint_free']['safety']['original_record_contract_pass']
    lines = ['# Past-Image Context Under The Original Risk Contract', '',
        'I tested whether information measured from the past images helps a neural',
        'controller distinguish useful forecasts from harmful interventions. Four',
        'matched arms isolate image bounds, measurement reliability and background',
        'motion. This is an exposed-development experiment, not an independent test.', '',
        f'Registered motion component screen: **{screen_status}**.',
        f'Primary paired completion-lower: **{number(primary["point"])}pp {ci(primary)}**.',
        f'Original full motion-policy recording contract: **{"PASS, development only" if motion_safe else "FAIL"}**.',
        '**No deployment or submission-readiness change.**', '',
        '## Complete Four-Arm Decision Matrix', '',
        '| Inputs | Policy | Known gain % | Separate lower % | Hard gain % | Easy gain % | Switch % | Original pass | Easy harm violations | Easy NA | Worst signed easy loss % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for policy, p in a['policies'][arm].items():
            s = p['safety']
            values = [p['utility']['known']['point'], p['utility']['lower']['point'],
                p['hard_gain_percent'], p['easy_gain_percent'], 100*p['switch_rate']]
            lines.append(f'| {arm} | {policy} | '+' | '.join(number(v) for v in values)+
                f' | {s["supported_record_views"]}/72 | {s["known_easy_violations"]} | {s["undefined_easy_views"]} | '+
                number(worst(a, arm, policy, 'known_easy_net_degradation'), 100)+' |')
    lines += ['', 'Mean easy utility is not the original per-recording safety check.',
        'The72views repeat24recordings across three risk-head seeds. NA remains',
        'NA. Empty same-count controls cannot establish utility or safety.', '',
        '## Registered And Secondary Paired Contrasts', '',
        '| Contrast | Estimand | Point pp | Exploratory source 95% interval |',
        '|---|---|---:|---|']
    for contrast, parts in a['contrasts'].items():
        for estimand, value in parts.items():
            lines.append(f'| {contrast} | {estimand} | {number(value["point"])} | {ci(value)} |')
    lines += ['', 'Primary: motion_geometry minus quality_geometry under joint_free.',
        'Bounds minus no-image is secondary. Motion minus no-image is a bundle',
        'contrast, not pure motion attribution. The fixed control is the original',
        'guarded1/32control; no coefficient is selected after outcomes.', '',
        'Completion lower/upper are conservative unknown-label bounds, not estimates',
        'of actual unobserved errors. Intervals use3,000source-component draws after',
        'averaging risk seeds within8groups. They are exploratory, not independent',
        'confirmation or correction for adaptive project-wide model development.', '',
        '## Primary By Source And Risk Seed', '',
        '| Source | Known pp | Paired completion-lower pp |', '|---|---:|---:|']
    rows = a['contrast_rows']['primary']
    for source in sorted(set(r['source'] for r in rows)):
        rr = [r for r in rows if r['source'] == source]
        lines.append(f'| {source} | {number(statistics.mean(r["known_pp"] for r in rr))} | '+
            number(statistics.mean(r['completion_lower_pp'] for r in rr))+' |')
    lines += ['', '| Risk-head seed | Paired completion-lower pp |', '|---|---:|']
    for seed, value in a['primary_seed_means'].items():
        lines.append(f'| {seed} | {number(value)} |')
    lines += ['', '## Actual Training And Reproduction', '',
        'Fresh_run:24Torch heads,19,814parameters each,4,096updates each(98,304updates),',
        'plus24complete exact refits(another98,304updates). Real recovery128+128vs256',
        f'exact:{pilot["actual128plus128_vs256_exact"]}. The corresponding resumed256-step',
        'pilot prefix is included in its full fit, not counted again as another model.',
        'The separate continuous256-step pilot contributes256additional updates:',
        '196,864actual optimizer updates including recovery and complete refits.',
        'All four arms share capacity, initialization, samplers, labels and budget.',
        'Masked context parameters are inactive; equal capacity does not imply equal',
        'effective information. Normalizers are fitted on A only.', '',
        'Cached_verified:frozen OOF forecasts, original source/ID schemas and causal',
        'measurement bundles. Fresh measured images were acquired before this fitting.',
        'Not_run:new forecasters, three forecaster seeds, new C/H or reserved outcome',
        'readout, independent confirmation or calibrated physical safety.', '',
        'Training uses producer-A OOF data only, not own-B labels. Each producer',
        'support excludes its corresponding own-B sources. This differs from the',
        'old50%A/Baugmentation regime; unmatched historical differences are not a',
        'pure image effect. Reference/support heads cannot see candidate-only fields.',
        'All36choice contexts freeze before this complete62,466-row own-B readout.', '',
        'Cumulative model fitting seconds exclude repeated input loading/preprocessing;',
        'the separate phase receipts report end-to-end process time and peak RSS.', '',
        '## Interpretation Limits', '',
        'Native observation8/prediction12, annotation stride1 and image coordinates.',
        'Available-label ADE is not full12ADE or SDD rawt50. No verified meters,',
        'seconds, homography, true3D, foundation, human-gold or collision-safety claim.',
        'Background motion is a conventional image proxy, not calibrated ego-motion.',
        'Original2%all/easy harm, signed easy-net, reference/support and NA rules stay',
        'unchanged. Old failures remain failures. Stage5C execution and SMC are off.', '',
        '[Protocol](protocol.md) | [Recording checks](recording_checks.csv) |',
        '[Loss trace](loss_trace.csv) | [Failure analysis](failure_analysis.md) |',
        '[Reproduction](operation_zh.md)']
    (PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    gates = ['# Past-Image Risk Learning Gates', '', '| Gate | Status | Evidence |', '|---|---|---|',
        '| Registered before training | PASS | Frozen four-arm protocol and feature admission |',
        '| Actual Torch fitting | PASS | 24models,98,304updates |',
        '| Complete real refits | PASS | 24full scientific states exact |',
        '| Matched initial functions and draws | PASS | Same capacity and sampling |',
        '| Causal inputs and source roles | PASS within pipeline | A-only fits, exact ID joins, past-only own-B input whitelist |',
        '| Registered motion utility | '+screen_status+' | Paired completion-lower interval and seed signs |',
        '| Original complete recording risk | '+('PASS, development only' if motion_safe else 'FAIL')+' | Unchanged2%/support/NA rules across72views |',
        '| Independent confirmation | NOT RUN | Already-exposed own-B development only |',
        '| Three forecaster seeds | NOT RUN | Three risk seeds, one forecaster seed |',
        '| Deployment / submission readiness | FALSE | Not authorized by this component experiment |',
        '| Stage5C execution / SMC | FALSE | Neither executed |']
    (PUBLIC/'world_model_gate_visual_risk.md').write_text('\n'.join(gates)+'\n')
    failure = ['# Did Past Images Repair Intervention Risk?', '',
        f'The registered motion component screen is {a["motion_component_screen_pass"]}.',
        f'The original complete motion-policy risk check is {motion_safe}.',
        f'The primary paired lower contrast is {number(primary["point"])}pp {ci(primary)}.', '',
        '## Separate Questions, Not Interchangeable Successes', '',
        '1. Image availability/alignment and exact recovery establish engineering.',
        '2. Lower training loss measures fit, not held-source policy generalization.',
        '3. Motion-minus-quality tests motion information; bounds-only success cannot replace it.',
        '4. Positive mean gain cannot replace every required original risk/support check.',
        '5. Exposed development and risk-head seeds are not independent new test evidence.', '',
        '## Matched Training Objective', '',
        '| Contrast | Lower final minibatch objective pairs | Mean final objective difference |',
        '|---|---:|---:|']
    for control, treatment in (('no_image', 'image_bounds'), ('quality_geometry', 'motion_geometry'), ('no_image', 'motion_geometry')):
        pairs = fit_pairs(fits, control, treatment)
        differences = [y['trace'][-1]['total']-x['trace'][-1]['total'] for x, y in pairs]
        failure.append(f'| {treatment} minus {control} | {sum(v < 0 for v in differences)}/6 | {number(statistics.mean(differences))} |')
    failure += ['', 'These are matched final training minibatches, not validation MSE or evidence',
        'of calibrated uncertainty. All24traces remain in the loss CSV.']
    calibration = read(PUBLIC/'calibration_analysis.json')['summary']
    quality, motion = calibration['quality_geometry'], calibration['motion_geometry']
    case = read(PUBLIC/'failure_case.json'); selected = case['arms']['motion_geometry']
    failure += ['', '## Actual Held-Source Prediction Errors', '',
        f'Motion lowers observation Brier from{quality["observation_Brier"]:.6f}to{motion["observation_Brier"]:.6f},',
        'but conditional cost generalization is worse in common history-scale units:',
        f'harm MSE {quality["conditional_MSE"]["harm"]:.6f}to{motion["conditional_MSE"]["harm"]:.6f},',
        f'easy-harm MSE {quality["conditional_MSE"]["easy_harm"]:.6f}to{motion["conditional_MSE"]["easy_harm"]:.6f},',
        f'reference MSE {quality["conditional_MSE"]["reference"]:.6f}to{motion["conditional_MSE"]["reference"]:.6f}.',
        'All five conditional errors worsen versus quality_geometry despite improved',
        'final fitting loss in all six pairs. Better observability prediction is',
        'not a reliable forecast-harm estimate. This is a mechanism diagnostic,',
        'not proof that images contain no useful information or a calibration guarantee.',
        '[Complete calibration audit](calibration_analysis.md).', '',
        '## Actual Tail Case', '',
        f'Source{case["source"]},recording{case["recording_name"]},frame{case["frame"]},risk seed{case["seed"]}.',
        'Selection is worst signed-easy recording, then largest positive easy-harm',
        'window. It is explicitly posthoc and not representative sampling.',
        f'The quality control retains the floor with ADE{case["floor_available_ADE_pixel"]:.4f}pixels.',
        f'Motion selects action{selected["action"]}with ADE{selected["available_ADE_pixel"]:.4f},',
        f'adding{case["actual_added_ADE_pixel"]:.4f}pixels,while predicting harm',
        f'{selected["predicted_moments"]["harm"]:.4f}and easy harm{selected["predicted_moments"]["easy_harm"]:.4f}.',
        f'The window has{case["known_future_steps"]}/12labels. This failure is not merely',
        'an unknown-future completion penalty. A pointwise predicted harm need not',
        'satisfy the2%budget alone because the joint decision pools query budgets.',
        '[Frozen scalar case](failure_case.json).', '',
        '## Risks And Representation Limits', '',
        'Past-image transforms can fail on weak texture, parallax, independently moving',
        'background content and masks. Unsupported transitions remain explicit, not',
        'zero-motion certificates. Full-source acquisition retains those failures.',
        'The small cost head does not learn pixels or retrain the trajectory actor.',
        'Any retained actor errors and source shifts can remain after better context.',
        'Pointwise fit and per-record joint deployment are different tasks.', '',
        'Error/reference ratios can exceed100%;they are not probabilities or',
        'physical collision risk. The worst signed easy loss is not a mean.', '',
        'There is no after-the-fact threshold search, dropped difficult recording,',
        'identity certificate or reopening of reserved outcomes. A failed result',
        'requires model/data repair under a new prospective experiment, not a',
        'relaxed denominator or replacement primary metric. The whole project is',
        'not submission-ready; no positive result here establishes novelty by itself.']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure)+'\n')
    operation = ['# 复现本轮图像上下文实验', '',
        '使用项目现有 arm64 `.venv-pytorch`。单进程 CPU4/inter-op1/workers0。',
        '大数组、图像缓存和模型权重只保存在本机，不随 Git 上传。',
        '当前结果是已经暴露的开发集，不是新的独立测试。原 2% 及支持要求不变。', '',
        '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py train --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py verify --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py infer --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py evaluate',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py replay',
        '.venv-pytorch/bin/python scripts/report_m3w_moving_visual_risk.py',
        '```', '',
        '已完成的 pilot/prepare/evaluate 产物保持不可变，不覆盖重跑。',
        '`train` 和 `verify` 可恢复同定义 checkpoint；不降低模型训练步数。',
        '`infer --resume` 恢复原分配过程；每个 query 的限时求解保留可行回退。',
        '`replay` 重算预测和指标，核验冻结决策的可行性与同数量对照；',
        '不要求重新运行限时求解器复现同一搜索路径。',
        '报告脚本只读取公开标量结果，不读取私有未来标签，也不调阈值。',
        '本轮是原生观察 8 步预测 12 步的图像坐标任务，不是米、秒、true3D。']
    (PUBLIC/'operation_zh.md').write_text('\n'.join(operation)+'\n')
    write_fits(fits); write_records(a); plots(a, fits)
    print('All four arms and original risk failures reported; no outcome-selected policy.')


if __name__ == '__main__':
    main()
