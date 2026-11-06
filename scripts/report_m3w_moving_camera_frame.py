"""Report complete frozen camera-frame actor evidence, without new scoring."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_camera_frame_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_camera_frame_v1'
SOURCE = ROOT/'outputs/publication_readiness_2026_09/moving_producer_crossfit_v1'


def read(path): return json.loads(path.read_text())
def fmt(x): return 'NA' if x is None else f'{x:+.4f}'
def mean(values): return None if not values or any(x is None for x in values) else statistics.mean(values)


def draw(a, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for arm, color in (('identity', '#565c64'), ('camera_frame', '#167d75')):
        traces = [read(SOURCE/'fits'/f'{r["job"]["key"]}.json')['result']['losses']
            if arm == 'identity' else r['result']['losses'] for r in fits]
        steps = sorted(set.intersection(*[set(x['step'] for x in trace) for trace in traces]))
        by_step = [{x['step']: x['loss'] for x in trace} for trace in traces]
        axes[0].plot(steps, [statistics.median(t[s] for t in by_step) for s in steps], color=color, label=arm)
    axes[0].set(xlabel='Optimizer updates', ylabel='Fitting objective, log scale', yscale='log',
        title='Six matched designs, one forecaster seed')
    axes[0].legend(); axes[0].grid(alpha=.2)
    rows = sorted((r for r in a['sources'] if r['contrast'] == 'primary'), key=lambda r: r['source'])
    known = [r['summary'][-1]['paired_known_pp'] for r in rows]
    lo = [r['summary'][-1]['paired_completion_lower_pp'] for r in rows]
    hi = [r['summary'][-1]['paired_completion_upper_pp'] for r in rows]
    axes[1].errorbar(range(14), known, yerr=[[v-l for v,l in zip(known, lo)], [h-v for v,h in zip(known, hi)]],
        fmt='o', color='#2d6a9f', capsize=3)
    axes[1].axhline(0, color='#565c64', linewidth=.7)
    axes[1].set_xticks(range(14), [r['source'].rsplit('_', 1)[-1] for r in rows], rotation=60)
    axes[1].set(xlabel='All source components', ylabel='Camera actor minus identity (pp)',
        title='Known effect and completion envelope, not CI')
    axes[1].grid(axis='y', alpha=.2)
    folder = PRIVATE/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'camera_training_and_sources.png', dpi=170); plt.close(fig)


def main():
    a = read(PUBLIC/'analysis.json'); p = a['primary']; pop = read(PUBLIC/'population.json')
    fits = [read(path) for path in sorted((PUBLIC/'fits').glob('*.json'))]
    training, verification, control = [read(PUBLIC/name) for name in ('training.json', 'verification.json', 'control.json')]
    if (len(fits) != 6 or training['models'] != 6 or verification['models'] != 1 or not verification['full_refits_exact']
            or not control['complete4000shared_state_and_common_trace_exact']
            or not all(r['result']['step'] == 4000 and r['result']['complete'] for r in fits)
            or not list((PRIVATE/'replays').glob('replay_*.json'))):
        raise ValueError('Complete actual full fits, control, full refit and exact replay required')
    primary = [r for r in a['sources'] if r['contrast'] == 'primary']
    original = read(ROOT/'outputs/publication_readiness_2026_09/moving_development_admission_v1/admission.json')
    coverage = {}
    for row in pop['roster']:
        source = original['recordings'][row['node']]['exclusion_group']
        totals = coverage.setdefault(source, [0, 0]); totals[0] += row['rows']
        totals[1] += row['rows'] if row['active'] else 0
    interval = p['interval95']; passed = p['component_progression_screen_pass']
    lines = ['# Does Explicit Past-Image Viewpoint Conditioning Help?', '',
        'I kept the same trajectory network and trained it in a query-view frame',
        'derived from past images. Predicted camera motion restores native output',
        'coordinates before loss. No new learned parameters or residual decoder.', '',
        f'Component screen:**{"PASS, exploratory" if passed else "FAIL"}**.',
        f'Whole12primary known difference:**{fmt(p["known_pp"])}pp**.',
        f'Paired completion lower:**{fmt(p["completion_lower_pp"])}pp**,exploratory95%',
        f'source-bootstrap interval**[{fmt(interval[0])},{fmt(interval[1])}]**.',
        f'Known-positive sources:{p["positive_known_sources"]}/14;producer-fold means:{p["fold_known_means"]}.', '',
        '**No new deployment,independent confirmation or complete selected-policy',
        'risk pass follows.** The original2%all/easy,signed easy-net,support/NA',
        'rules remain. A failed lower bound does not identify the actual unknown',
        'effect. The experiment uses exposed A donors,not a new test.', '',
        '## Matched Prefix Comparisons', '',
        '| Contrast | Prefix | Control gain vs CV % | New gain vs CV % | Paired known pp | Lower pp | Upper pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for contrast in ('primary', 'camera_cv_effect', 'neural_over_camera_cv'):
        rows = [r for r in a['sources'] if r['contrast'] == contrast]
        for i, prefix in enumerate((1, 3, 6, 12)):
            group = [r['summary'][i] for r in rows]
            values = [mean([x['control']['known_gain_percent'] for x in group]),
                mean([x['treatment']['known_gain_percent'] for x in group])]+[
                mean([x[k] for x in group]) for k in ('paired_known_pp', 'paired_completion_lower_pp', 'paired_completion_upper_pp')]
            lines.append(f'| {contrast} | {prefix} | '+' | '.join(fmt(x) for x in values)+' |')
    lines += ['', 'Primary is the camera-conditioned actor minus the cached identity actor.',
        'Camera-CV minus raw CV and neural minus camera-CV are secondary controls.',
        'A useful camera baseline alone is not proof of neural added value.',
        'Weights balance queries within each source,then sources. Missing references',
        'remain NA;no defined-only averaging or outcome-selected primary.', '',
        '## Causal Baseline Diagnostics', '',
        '| Baseline | Prefix | Known gain vs CV % | Hard gain % | Easy positive harm % |',
        '|---|---:|---:|---:|---:|']
    for name in sorted({r['baseline'] for r in a['causal_baselines']}):
        rows = [r for r in a['causal_baselines'] if r['baseline'] == name]
        for i, prefix in enumerate((1, 3, 6, 12)):
            values = [mean([r['summary'][i]['treatment'][k] for r in rows]) for k in
                ('known_gain_percent', 'known_hard_gain_percent', 'known_easy_positive_harm_percent')]
            lines.append(f'| {name} | {prefix} | '+' | '.join(fmt(x) for x in values)+' |')
    lines += ['', 'All seven fixed causal baselines are reported. An aggregate best score',
        'on these exposed donors is diagnostic,not a new selected deployment floor.', '',
        '## Every Source, Whole12', '',
        '| Source | Rows | Active camera rows | Known | Partial | Unknown | Known pp | Lower pp | Upper pp | New easy +harm % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(primary, key=lambda r: r['source']):
        s = row['summary'][-1]; total, active = coverage[row['source']]
        if total != s['rows']: raise ValueError('Report the complete original source population')
        lines.append(f'| {row["source"]} | {total} | {active} | {s["known_rows"]} | {s["partial_rows"]} | {s["wholly_unknown_rows"]} | '+
            ' | '.join(fmt(x) for x in (s['paired_known_pp'], s['paired_completion_lower_pp'],
                s['paired_completion_upper_pp'], s['treatment']['known_easy_positive_harm_percent']))+' |')
    harm = [r['summary'][-1]['treatment']['known_easy_positive_harm_percent'] for r in a['recordings'] if r['contrast'] == 'primary']
    lines += ['', f'Raw camera actor known-easy +harm exceeds2%in{sum(x is not None and x > 2 for x in harm)}/34recordings;',
        f'{sum(x is None for x in harm)}undefined easy references remain NA.',
        'This is a raw actor diagnostic,not a complete selected-policy certificate.', '',
        '## Camera Support Strata, Whole12', '',
        '| Fitting design | Camera active | Rows | Contrast | Known pp | Lower pp |',
        '|---|---|---:|---|---:|---:|']
    for row in a['camera_strata']:
        s = row['summary'][-1]
        lines.append(f'| {row["job"]} | {row["camera_active"]} | {s["rows"]} | {row["contrast"]} | {fmt(s["paired_known_pp"])} | {fmt(s["paired_completion_lower_pp"])} |')
    lines += ['', 'Active and unsupported populations differ;their separate means do not',
        'identify a causal effect of camera-support status. No slice is removed.', '',
        '## What Actually Ran', '',
        'Fresh_run:six complete4000update treatment fits,one full identity-control',
        'refit and one full first-design treatment refit. Not six full refits.',
        'Real100+100vs200recovery is exact;its200prefix seeds its own treatment.',
        'Direct200is additional compute. Uninterrupted real-data budget32,200updates.',
        'A later tool interruption required resuming the fifth treatment from',
        'update1400,last observed heartbeat1550. At least150updates were repeated;',
        'the exact discarded count and old process exit code were not observed.',
        '32,200is therefore the planned budget,not a measured total including',
        'that interruption. The four completed fits were reused unchanged.',
        'All6actor and6camera-CV prediction banks freeze before scoring;all original',
        'predictions and scalars reproduce exactly. The generic loader opens/hashes',
        'target arrays for integrity,but targets/masks do not enter model inputs.', '',
        f'Fresh causal composition retains{pop["rows"]}rows,{pop["queries"]}queries.',
        f'{pop["active_rows"]}rows use complete seven-pair support;all remaining rows',
        'remain with explicit identity imputation. That is not measured zero camera',
        'motion or a frozen raw-model fallback. The newly trained weights can change',
        'unsupported forecasts. No donor/test statistics normalize model inputs.', '',
        'Cached_verified:the six raw controls,source split,packed geometry and',
        'previously measured past image-affine proxies. Not_run:new own-B/C/H',
        'readout,new controller fitting,three forecaster seeds or independent',
        'confirmation. Conventional viewpoint conditioning is not a new method',
        'contribution by itself and does not inherit a metric/equivariance proof.', '',
        'Nativearm64CPU4/inter-op1/workers0,full4000endpoint,checkpoint/heartbeat/',
        'resume,no quick downgrade or threshold search. Native observation8/',
        'prediction12,stride1,pixel coordinates,not SDDt50,meters,verified seconds,',
        'true3D,foundation,human-gold or physical-safety success. Stage5C/SMCoff.', '',
        '[Protocol](protocol.md) | [Model scope](model_scope.md) | [Causal inputs](input_operator_report.md) |',
        '[Loss trace](loss_trace.csv) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    fields = ('job', 'step', 'loss', 'gradient_norm', 'learning_rate', 'supported_batch_rows', 'mean_past_normalized_ADE')
    with (PUBLIC/'loss_trace.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for row in fits:
            for loss in row['result']['losses']: writer.writerow(dict(job=row['job']['key'], **loss))
    camera_effect = mean([r['summary'][-1]['paired_known_pp'] for r in a['sources'] if r['contrast'] == 'camera_cv_effect'])
    neural_effect = mean([r['summary'][-1]['paired_known_pp'] for r in a['sources'] if r['contrast'] == 'neural_over_camera_cv'])
    failure = ['# Geometric Actor Result And Remaining Gap', '',
        f'Primary known:{fmt(p["known_pp"])}pp;paired lower:{fmt(p["completion_lower_pp"])}pp.',
        f'Component screen:{passed}. Camera-CV effect:{fmt(camera_effect)}known pp.',
        f'Neural minus camera-CV:{fmt(neural_effect)}known pp.', '',
        'These controls distinguish simple geometric forecasting from additional',
        'learned forecasting. They do not isolate true camera pose,depth,scene',
        'semantics or the causes of every source failure. Constant-affine future',
        'extrapolation can fail under changing viewpoint,parallax or measurement',
        'error. Those are hypotheses unless the retained slices support them.', '',
        'A negative completion bound alone does not prove the unknown effect is',
        'negative. Known,partial and wholly unknown outcomes stay distinct.',
        'The source-fold contrasts and camera support slices remain visible.',
        'All source rows remain;no primary/threshold is changed after scoring.', '',
        'No new controller is trained and no new policy is deployed. The original',
        'per-recording2%all/easy,signed easy-net,reference/support/NA requirements',
        'remain. Exact replay is reproducibility evidence,not independent',
        'confirmation. Three forecaster seeds and compliant new confirmation',
        'remain missing. Prior image-feature and prefix-loss failures are not',
        'overwritten. The paper is not submission-ready.']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure)+'\n')
    gates = ['# Camera-Frame Component Gates', '', '| Gate | Status |', '|---|---|',
        '| Actual source-held full Torch fitting | PASS:6treatments4000each |',
        '| Full original-control parity | PASS |', '| Real-data100+100vs200recovery pilot | PASS:full scientific state exact |',
        '| Later tool interruption | Same fifth checkpoint resumed;exact discarded compute unknown |',
        '| Full treatment refit | PASS:first fixed design only |',
        '| All predictions/scalars replay | PASS |',
        '| Whole12component screen | '+('PASS,exploratory' if passed else 'FAIL')+' |',
        '| New full selected-policy risk | NOT RUN |', '| Independent confirmation/three forecaster seeds | NOT RUN |',
        '| Deployment/submission readiness | FALSE |', '| Stage5C execution/SMC | FALSE |']
    (PUBLIC/'world_model_gate_camera_frame.md').write_text('\n'.join(gates)+'\n')
    commands = ['# 本轮视角条件化轨迹模型复现', '',
        '使用 arm64 `.venv-pytorch`，CPU4/inter-op1/workers0。需要现有已校验',
        'native轨迹、过去图像变换代理和 pinned EqMotion 源码。原始数据、缓存、',
        '第三方源码及checkpoint不在Git。不要新建重复训练进程。', '', '```sh']
    for phase in ('control --resume', 'train --resume', 'verify --resume', 'forecast', 'evaluate', 'replay'):
        commands.append('.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py '+phase)
    commands += ['.venv-pytorch/bin/python scripts/report_m3w_moving_camera_frame.py', '```', '',
        '已完成的 pilot/evaluate 收据不能覆盖；replay 复算预测和原始指标。',
        '训练使用同定义checkpoint恢复，不减少4000步，也不改变风险门槛。',
        '完整重训核验仅一个固定design，不能称六个都完整重训。',
        'Camera extrapolation只来自过去七对图像，未来真实相机位姿不作输入。',
        'loss CSV 的 mean_past_normalized_ADE 是沿用字段名，数值实际为未加权',
        'batch available-label ADE的packed图像坐标值，不是米或统一跨场景尺度。',
        '这是训练侧来源排除组件实验，不是新独立测试或完整部署认证。']
    (PUBLIC/'operation_zh.md').write_text('\n'.join(commands)+'\n')
    draw(a, fits)
    print('Complete camera-frame component evidence reported;no deployment claim.')


if __name__ == '__main__': main()
