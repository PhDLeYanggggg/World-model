"""Report the complete frozen native/observed-view node-encoder comparison."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = HOME/'moving_dual_view_actor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_dual_view_actor_v1'


def read(path): return json.loads(path.read_text())
def fmt(x): return 'NA' if x is None else f'{x:+.4f}'
def mean(values): return None if not values or any(x is None for x in values) else statistics.mean(values)


def draw(a, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for arm, color in (('raw', '#565c64'), ('old_hard_restore', '#9c5c7e'), ('native_node', '#2d6a9f'), ('camera_node', '#167d75')):
        if arm in ('raw', 'old_hard_restore'):
            folder = HOME/('moving_producer_crossfit_v1' if arm == 'raw' else 'moving_camera_frame_v1')/'fits'
            traces = [read(path)['result']['losses'] for path in sorted(folder.glob('*.json'))]
        else: traces = [r['result']['losses'] for r in fits if r['arm'] == arm]
        if len(traces) != 6: raise ValueError('Six complete designs per trace required')
        steps = sorted(set.intersection(*[set(x['step'] for x in trace) for trace in traces]))
        lookup = [{x['step']: x['loss'] for x in trace} for trace in traces]
        axes[0].plot(steps, [statistics.median(t[s] for t in lookup) for s in steps], color=color, label=arm)
    axes[0].set(xlabel='Optimizer updates', ylabel='Fitting objective, log scale', yscale='log',
        title='Matched designs; old controls are verified cache')
    axes[0].legend(); axes[0].grid(alpha=.2)
    rows = sorted((r for r in a['sources'] if r['contrast'] == 'primary'), key=lambda r: r['source'])
    k = [r['summary'][-1]['paired_known_pp'] for r in rows]
    lo = [r['summary'][-1]['paired_completion_lower_pp'] for r in rows]
    hi = [r['summary'][-1]['paired_completion_upper_pp'] for r in rows]
    axes[1].errorbar(range(14), k, yerr=[[x-l for x,l in zip(k, lo)], [h-x for x,h in zip(k, hi)]], fmt='o', capsize=3, color='#167d75')
    axes[1].set_xticks(range(14), [r['source'].rsplit('_', 1)[-1] for r in rows], rotation=60)
    axes[1].axhline(0, color='#565c64', linewidth=.7); axes[1].grid(axis='y', alpha=.2)
    axes[1].set(xlabel='Every source component', ylabel='Camera node minus native node (pp)',
        title='Known effect and completion envelope, not CI')
    folder = PRIVATE/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'dual_view_training_and_sources.png', dpi=170); plt.close(fig)


def main():
    a = read(PUBLIC/'analysis.json'); p, bundle = a['primary'], a['camera_bundle']
    fits = [read(path) for path in sorted((PUBLIC/'fits').glob('*/*.json'))]
    t, v, c = [read(PUBLIC/name) for name in ('training.json', 'verification.json', 'control.json')]
    if (len(fits) != 12 or t['models'] != 12 or v['models'] != 2 or not v['full_refits_exact']
            or not c['shared_core_optimizer_sampler_RNG_and_common_trace_exact']
            or not all(r['result']['complete'] and r['result']['step'] == 4000 for r in fits)
            or not list((PRIVATE/'replays').glob('replay_*.json')) or not a['parent_both_and_baseline_statistics_exact']):
        raise ValueError('Complete full matrix,shared control,two full refits and exact replay required')
    passed = a['component_progression_screen_pass']
    lines = ['# Does Additional Per-Node Observed-View Motion Help?', '',
        'I retain native geometry and supply aligned histories as additional node',
        'features. There is no hard future-camera restoration or trajectory residual',
        'decoder. The matched native-node control has the same extra capacity.', '',
        f'Overall component screen:**{"PASS,exploratory" if passed else "FAIL"}**.', '',
        '| Fixed comparison | Known pp | Completion lower pp | Exploratory95%source interval | Positive sources | Screen |',
        '|---|---:|---:|---|---:|---|']
    for label, result in (('Primary:camera-native node', p), ('Required camera-raw bundle', bundle)):
        lo, hi = result['interval95']
        lines.append(f'| {label} | {fmt(result["known_pp"])} | {fmt(result["completion_lower_pp"])} | [{fmt(lo)},{fmt(hi)}] | '+
            f'{result["positive_known_sources"]}/14 | {result["component_progression_screen_pass"]} |')
    lines += ['', f'Primary fold known means:{p["fold_known_means"]};camera/raw:{bundle["fold_known_means"]}.',
        'Both screens are required;beating a worse capacity control alone cannot',
        'license controller fitting. No new policy,test readout or deployment.', '',
        '## Every Fixed Contrast And Prefix', '',
        '| Contrast | Prefix | Control gain vs CV % | New gain vs CV % | Paired known pp | Lower pp | Upper pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for contrast in ('primary', 'camera_bundle', 'native_bundle', 'both_effect', 'camera_over_camera_cv', 'camera_cv_effect'):
        rows = [r for r in a['sources'] if r['contrast'] == contrast]
        if len(rows) != 14: raise ValueError('Retain every source in every contrast')
        for i, prefix in enumerate((1, 3, 6, 12)):
            vals = [mean([r['summary'][i][side]['known_gain_percent'] for r in rows]) for side in ('control', 'treatment')]
            vals += [mean([r['summary'][i][key] for r in rows]) for key in ('paired_known_pp', 'paired_completion_lower_pp', 'paired_completion_upper_pp')]
            lines.append(f'| {contrast} | {prefix} | '+' | '.join(fmt(x) for x in vals)+' |')
    lines += ['', 'The primary is matched camera-node minus native-node. Camera/raw and',
        'native/raw are extra-capacity bundle contrasts. Old hard-restoration',
        'and camera-CV controls are cached_verified,not fresh model fits. Their',
        'readout and all seven causal baseline statistics match the parent exactly.',
        'Means balance queries within source then all sources;NA is not dropped.', '',
        '## Every Source,Primary Whole12', '',
        '| Source | Rows | Known | Partial | Unknown | Known pp | Lower pp | New easy +harm % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted((r for r in a['sources'] if r['contrast'] == 'primary'), key=lambda r: r['source']):
        s = r['summary'][-1]
        lines.append(f'| {r["source"]} | {s["rows"]} | {s["known_rows"]} | {s["partial_rows"]} | {s["wholly_unknown_rows"]} | '+
            ' | '.join(fmt(x) for x in (s['paired_known_pp'], s['paired_completion_lower_pp'], s['treatment']['known_easy_positive_harm_percent']))+' |')
    harms = [r['summary'][-1]['treatment']['known_easy_positive_harm_percent'] for r in a['recordings'] if r['contrast'] == 'primary']
    lines += ['', f'Raw camera-node known-easy +harm exceeds2%in{sum(x is not None and x > 2 for x in harms)}/34recordings;',
        f'{sum(x is None for x in harms)}undefined references remain NA. This is not a full policy certificate.',
        'All recordings,camera support strata and baseline diagnostics remain in',
        'the scalar receipt;no favorable prefix or source replaces the primary.', '',
        '## Actual Execution And Limits', '',
        'Fresh_run:twelve full4000update treatment models,one full frozen-encoder',
        'raw control and two complete first-design refits,one per learning arm.',
        'Original shared core,active optimizer state,sampler/RNG/factors/common',
        'trace match the old control. Two extra inactive optimizer parameter entries',
        'are expected,not identical complete parameter lists. Both learning arms',
        'have3,029,060nominal/582,468trainable parameters,1,792extra node parameters.', '',
        'Real100+100and actual in-forward SIGTERM1+199both match continuous200.',
        'The immutable step1stop snapshot is preserved. Three recovery trajectories',
        'add400updates beyond the prefix reused by its own treatment;planned',
        'uninterrupted total60,400updates. Any repeated interrupted work must be',
        'accounted separately. A partial-update failure retains a coherent earlier',
        'checkpoint;it never certifies an advanced incomplete sampler as committed.', '',
        'All twelve new prediction banks freeze before scoring and replay exactly;',
        'all scalar evidence replays exactly. Eighteen cached raw/old-both/camera-CV',
        'banks are hash/row-alignment verified,not freshly inferred here. Generic',
        'loading opens/hashes label arrays for integrity;targets/masks enter only',
        'loss/evaluation. No future-camera operator,endpoint,central velocity or',
        'test endpoint goal enters the model. Native geometry is always retained.', '',
        'Previously exposed125540producer-Awindows,14sources,34recordings,1613queries.',
        'One forecaster seed17. Native observation8/prediction12,stride1,image',
        'coordinates;not SDDt50,verified seconds,meters,homography,true3D,foundation,',
        'human-gold labels or physical-safety success. Original per-recording2%all/',
        'easy positive-harm,signed easy-net,reference/support/NA rules unchanged.',
        'No identity certification,epsilon,new own-B/C/Hreadout,new controller or',
        'deployment. A two-view encoder alone is not a new method contribution.',
        'Formal three-forecaster-seed main results,independent confirmation and',
        'full selected-policy pass remain not_run. Submission not ready;Stage5C/',
        'SMCoff. Earlier completed failures and the unscored stopped trial remain.', '',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Loss traces](loss_trace.csv) | [Reproduction](operation_zh.md)']
    (PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    fields = ('arm', 'job', 'step', 'loss', 'gradient_norm', 'learning_rate', 'supported_batch_rows', 'mean_past_normalized_ADE')
    with (PUBLIC/'loss_trace.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for fit in fits:
            for loss in fit['result']['losses']: writer.writerow(dict(arm=fit['arm'], job=fit['job']['key'], **loss))
    failure = ['# Node-View Result And Remaining Model Gap', '',
        f'Primary known{fmt(p["known_pp"])}pp/lower{fmt(p["completion_lower_pp"])}pp.',
        f'Camera/raw known{fmt(bundle["known_pp"])}pp/lower{fmt(bundle["completion_lower_pp"])}pp.',
        f'Overall component screen:{passed}.', '',
        'The native-node contrast controls extra encoder capacity;the raw contrast',
        'checks that the new branch is useful beyond the original forecaster.',
        'Both are necessary. No secondary result can replace a failed primary.',
        'A negative completion lower is not identification of unknown actual loss.', '',
        'Native geometry is retained,so the earlier one-sided information-erasure',
        'counterexample is repaired at the representation level. That is not proof',
        'that the proxy contains useful camera/scene semantics on real held sources.',
        'Current node features use observed velocity/displacement,not semantic pixels,',
        'verified ego pose or a learned future-camera model. They can be noisy under',
        'parallax or track/image measurement errors. Those mechanisms are hypotheses',
        'unless demonstrated by retained diagnostics. A new network or a lower',
        'fitting objective alone does not demonstrate reliable intervention.', '',
        'No new controller or deployment. Original2%/support/NA contract remains.',
        'One forecaster seed and previously exposed sources are not independent',
        'confirmation. Full model/data release and paper-level evidence are still',
        'missing. Stage5C execution and SMCoff.']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure)+'\n')
    gates = ['# Information-Preserving Node-View Gates', '', '| Gate | Status |', '|---|---|',
        '| Full actual Torch matrix | PASS:12models4000each |',
        '| Shared original-core/active-optimizer parity | PASS:extra inactive entries declared |',
        '| Real normal and mid-forward signal recovery | PASS:both exact against continuous200 |',
        '| Two complete treatment refits | PASS:first fixed design per arm,not all twelve |',
        '| Twelve new predictions/all scalars replay | PASS |',
        '| Matched-capacity primary screen | '+str(p['component_progression_screen_pass'])+' |',
        '| Additional camera/raw screen | '+str(bundle['component_progression_screen_pass'])+' |',
        '| Overall component progression | '+str(passed)+' |',
        '| Full policy risk/independent confirmation/three forecaster seeds | NOT RUN |',
        '| Deployment/submission readiness/Stage5C execution/SMC | FALSE |']
    (PUBLIC/'world_model_gate_dual_view.md').write_text('\n'.join(gates)+'\n')
    commands = ['# 原始与对齐双视图轨迹模型复现', '',
        '需要已有 native 数据、过去图像变换代理和 pinned EqMotion。使用 arm64',
        '`.venv-pytorch`，CPU4/inter-op1/workers0。不要重复启动活跃进程。', '', '```sh']
    for phase in ('admit', 'control --resume', 'train --resume', 'verify --resume', 'forecast', 'evaluate', 'replay'):
        commands.append('.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py '+phase)
    commands += ['.venv-pytorch/bin/python scripts/report_m3w_moving_dual_view.py', '```', '',
        '已完成pilot/evaluate收据不可覆盖；replay重算12个新预测及全部指标。',
        '原18个对照本轮只校验hash与逐行ID，不说成新推理。原始几何始终保留，',
        '模型输入不含未来相机外推。中途信号在完整更新后停止；异常的半步状态',
        '不会覆盖最后一致checkpoint。不要复用旧单边实验的半步checkpoint。',
        'loss字段mean_past_normalized_ADE沿用旧名称，实际是batch未加权',
        'available-label ADE的packed图像坐标数值，不是米或统一跨域尺度。',
        '两个完整重训核验只覆盖每个arm固定第一design。不是独立测试或部署认证。']
    (PUBLIC/'operation_zh.md').write_text('\n'.join(commands)+'\n')
    draw(a, fits)
    print('Completed dual-view component evidence reported;no deployment claim.')


if __name__ == '__main__': main()
