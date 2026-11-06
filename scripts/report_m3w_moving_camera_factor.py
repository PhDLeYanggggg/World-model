"""Report completed operator ablations from frozen scalar receipts only."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = HOME/'moving_camera_factor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_camera_factor_v1'


def read(path): return json.loads(path.read_text())
def fmt(x): return 'NA' if x is None else f'{x:+.4f}'
def mean(values): return None if not values or any(x is None for x in values) else statistics.mean(values)


def figures(a, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for arm, color in (('identity', '#565c64'), ('both', '#9c5c7e'), ('past_only', '#167d75'), ('future_only', '#2d6a9f')):
        if arm in ('identity', 'both'):
            folder = HOME/('moving_producer_crossfit_v1' if arm == 'identity' else 'moving_camera_frame_v1')/'fits'
            traces = [read(path)['result']['losses'] for path in sorted(folder.glob('*.json'))]
        else:
            traces = [row['result']['losses'] for row in fits if row['arm'] == arm]
        if len(traces) != 6: raise ValueError('Six complete traces per arm required')
        steps = sorted(set.intersection(*[set(x['step'] for x in trace) for trace in traces]))
        lookup = [{x['step']: x['loss'] for x in trace} for trace in traces]
        axes[0].plot(steps, [statistics.median(t[s] for t in lookup) for s in steps], label=arm, color=color)
    axes[0].set(xlabel='Optimizer updates', ylabel='Fitting objective, log scale', yscale='log',
        title='Six matched designs per arm, one forecaster seed')
    axes[0].grid(alpha=.2); axes[0].legend()
    rows = sorted((r for r in a['sources'] if r['contrast'] == 'primary'), key=lambda r: r['source'])
    known = [r['summary'][-1]['paired_known_pp'] for r in rows]
    low = [r['summary'][-1]['paired_completion_lower_pp'] for r in rows]
    high = [r['summary'][-1]['paired_completion_upper_pp'] for r in rows]
    axes[1].errorbar(range(14), known,
        yerr=[[k-l for k,l in zip(known, low)], [h-k for k,h in zip(known, high)]],
        fmt='o', capsize=3, color='#167d75')
    axes[1].set_xticks(range(14), [r['source'].rsplit('_', 1)[-1] for r in rows], rotation=60)
    axes[1].axhline(0, color='#565c64', linewidth=.7); axes[1].grid(axis='y', alpha=.2)
    axes[1].set(xlabel='Every source component', ylabel='Past-only minus identity (pp)',
        title='Known effect and completion envelope, not CI')
    folder = PRIVATE/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'factor_training_and_sources.png', dpi=170); plt.close(fig)


def main():
    a = read(PUBLIC/'analysis.json'); p = a['primary']; reg = read(PUBLIC/'registration.json')
    fits = [read(path) for path in sorted((PUBLIC/'fits').glob('*/*.json'))]
    trained, verified, control = [read(PUBLIC/name) for name in ('training.json', 'verification.json', 'control.json')]
    replays = list((PRIVATE/'replays').glob('replay_*.json'))
    if (len(fits) != 12 or trained['models'] != 12 or verified['models'] != 2 or not verified['full_refits_exact']
            or not control['complete4000shared_state_and_common_trace_exact'] or not replays
            or not all(r['result']['step'] == 4000 and r['result']['complete'] for r in fits)
            or not a['parent_both_and_baseline_statistics_exact']):
        raise ValueError('Complete full matrix, original control, two full refits and exact replay required')
    passed = p['component_progression_screen_pass']; ci = p['interval95']
    lines = ['# Does Past Alignment Help Without Hard Future-Camera Restoration?', '',
        'I separated the previous two-operator repair into a fully retrained',
        'four-arm comparison. The trajectory network,loss,source exclusions and',
        '4,000-update budget stay fixed. There are no new learned parameters.', '',
        f'Primary component screen:**{"PASS,exploratory" if passed else "FAIL"}**.',
        f'Past-only minus identity whole12known effect:**{fmt(p["known_pp"])}pp**.',
        f'Paired completion lower:**{fmt(p["completion_lower_pp"])}pp**,exploratory95%',
        f'source-bootstrap interval**[{fmt(ci[0])},{fmt(ci[1])}]**.',
        f'Known-positive sources:{p["positive_known_sources"]}/14;fold means:{p["fold_known_means"]}.', '',
        'The primary is fixed past_only minus identity,not whichever arm looks',
        'best after scoring. Secondary effects cannot replace it. No new controller,',
        'test access or deployment follows from this component report.', '',
        '## Four Arms,Same Source And Query Weights', '',
        '| Arm | Past aligned | Future restored | Prefix | Known gain vs CV % | Hard gain % | Easy +harm % |',
        '|---|---|---|---:|---:|---:|---:|']
    arm_lookup = {'identity': ('primary', 'control'), 'past_only': ('primary', 'treatment'),
        'future_only': ('future_effect', 'treatment'), 'both': ('both_effect', 'treatment')}
    for arm, (contrast, field) in arm_lookup.items():
        rows = [r for r in a['sources'] if r['contrast'] == contrast]
        for i, prefix in enumerate((1, 3, 6, 12)):
            vals = [mean([r['summary'][i][field][k] for r in rows]) for k in
                ('known_gain_percent', 'known_hard_gain_percent', 'known_easy_positive_harm_percent')]
            lines.append(f'| {arm} | {arm in ("past_only", "both")} | {arm in ("future_only", "both")} | {prefix} | '+
                ' | '.join(fmt(x) for x in vals)+' |')
    lines += ['', '## All Paired Contrasts', '',
        '| Contrast | Prefix | Known pp | Completion lower pp | Completion upper pp |',
        '|---|---:|---:|---:|---:|']
    for contrast in ('primary', 'remove_future', 'future_effect', 'add_past_to_future',
            'both_effect', 'past_over_camera_cv', 'camera_cv_effect'):
        rows = [r for r in a['sources'] if r['contrast'] == contrast]
        if len(rows) != 14: raise ValueError('Do not omit a source from any contrast')
        for i, prefix in enumerate((1, 3, 6, 12)):
            vals = [mean([r['summary'][i][k] for r in rows]) for k in
                ('paired_known_pp', 'paired_completion_lower_pp', 'paired_completion_upper_pp')]
            lines.append(f'| {contrast} | {prefix} | '+' | '.join(fmt(x) for x in vals)+' |')
    interaction = [mean([r['known_pp_by_prefix'][i] for r in a['descriptive_known_interaction']]) for i in range(4)]
    lines += ['', f'Descriptive known-label interaction at1/3/6/12steps:{[fmt(x) for x in interaction]}.',
        'It is(both-future_only)-(past_only-identity),not an identified unknown-',
        'label interaction or a deployment test. No unrelated bound subtraction.', '',
        '## Every Source,Primary Whole12', '',
        '| Source | Rows | Known pp | Lower pp | Upper pp | Past-only easy +harm % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in sorted((r for r in a['sources'] if r['contrast'] == 'primary'), key=lambda r: r['source']):
        s = r['summary'][-1]
        lines.append(f'| {r["source"]} | {s["rows"]} | '+ ' | '.join(fmt(x) for x in
            (s['paired_known_pp'], s['paired_completion_lower_pp'], s['paired_completion_upper_pp'],
             s['treatment']['known_easy_positive_harm_percent']))+' |')
    lines += ['', 'All sources,recordings and supported/unsupported-camera strata remain',
        'in the scalar receipt. Unsupported histories retain identity operators,',
        'not a frozen-model safety fallback. Their trained predictions can change.', '',
        '## Raw Actor Recording Diagnostics', '', '| Arm | Known-easy +harm over2% /34 | Undefined easy references |', '|---|---:|---:|']
    harms = {}
    for arm, (contrast, field) in arm_lookup.items():
        values = [r['summary'][-1][field]['known_easy_positive_harm_percent'] for r in a['recordings'] if r['contrast'] == contrast]
        harms[arm] = sum(x is not None and x > 2 for x in values)
        lines.append(f'| {arm} | {harms[arm]} | {sum(x is None for x in values)} |')
    lines += ['', 'Raw-actor checks are not the complete selected-policy contract. Original',
        'per-recording2%all/easy positive-harm,signed easy-net,reference/support/NA',
        'requirements remain. No identity certificate,epsilon or defined-only mean.', '',
        '## Causal Baseline Diagnostics,Whole12', '',
        '| Baseline | Known gain vs CV % | Hard gain % | Easy +harm % |', '|---|---:|---:|---:|']
    for baseline in sorted({r['baseline'] for r in a['causal_baselines']}):
        rows = [r for r in a['causal_baselines'] if r['baseline'] == baseline]
        vals = [mean([r['summary'][-1]['treatment'][k] for r in rows]) for k in
            ('known_gain_percent', 'known_hard_gain_percent', 'known_easy_positive_harm_percent')]
        lines.append(f'| {baseline} | '+' | '.join(fmt(x) for x in vals)+' |')
    lines += ['', 'Original both-arm and seven-baseline statistics match the prior experiment',
        'exactly. Recomputing those cached outcomes is not new independent evidence.', '',
        '## What Ran And What Did Not', '',
        'Fresh_run:twelve full4000update treatment fits,one full identity control',
        'and two full fixed-first-design treatment refits,one per new arm. Not',
        'twelve full refits or three forecaster seeds. Real100+100vs200recovery',
        'matches complete scientific state;its200prefix seeds its own treatment.',
        'Planned uninterrupted optimizer budget60,200includes additional direct200.',
        'Any interrupted/repeated updates must be accounted separately.', '',
        'All twelve new forecast banks freeze before scoring and replay exactly;',
        'all reported scalars replay exactly. The eighteen cached identity/both/',
        'camera-CV banks are hash/row-alignment verified,not freshly inferred here.',
        'Existing operators/geometry are reused without copying the raw dataset.', '',
        'Not_run:new own-B/C/H/reserved outcomes,new controller,independent',
        'confirmation,formal three-forecaster-seed main study or full policy pass.',
        'Generic loading opens/hashes labels for integrity;targets/masks never',
        'enter inference. No future endpoint,central velocity or test endpoint goals.', '',
        'Nativearm64CPU4/inter-op1/workers0,checkpoint/heartbeat/resume,no quick',
        'downgrade. Previously exposed125540Arows,14sources,34recordings,1613queries.',
        'Observation8/prediction12,stride1,image coordinates,not SDDt50,verified',
        'seconds,meters,true3D,foundation,human-gold or physical-safety success.',
        'A conventional operator ablation is not a new method contribution by',
        'itself. Submission remains not ready. Stage5C execution and SMCoff.', '',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Gates](world_model_gate_camera_factor.md) |',
        '[Loss traces](loss_trace.csv) | [Reproduction](operation_zh.md)']
    (PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    fields = ('arm', 'job', 'step', 'loss', 'gradient_norm', 'learning_rate', 'supported_batch_rows', 'mean_past_normalized_ADE')
    with (PUBLIC/'loss_trace.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for fit in fits:
            for loss in fit['result']['losses']: writer.writerow(dict(arm=fit['arm'], job=fit['job']['key'], **loss))
    losses = []
    for fit in fits:
        original = read(HOME/'moving_producer_crossfit_v1/fits'/f'{fit["job"]["key"]}.json')['result']['losses'][-1]['loss']
        losses.append((fit['arm'], fit['job']['key'], original, fit['result']['losses'][-1]['loss']))
    failure = ['# Operator-Ablation Failure Analysis', '',
        f'Primary known:{fmt(p["known_pp"])}pp;paired lower:{fmt(p["completion_lower_pp"])}pp;screen:{passed}.', '',
        'The primary remains past_only minus identity. Removing future restoration',
        'can beat the old both-arm failure without beating the original actor;',
        'those are different questions. All secondary contrasts are retained.',
        'A negative completion lower does not identify the unknown actual effect.', '',
        '## Matched Final Fitting Minibatches', '',
        '| Arm | Design | Identity loss | New loss |', '|---|---|---:|---:|']
    failure += [f'| {arm} | {job} | {old:.6f} | {new:.6f} |' for arm, job, old, new in losses]
    failure += ['', 'These are final matched minibatches,not full training-population averages.',
        'The four retrained/cached arms separate total operator-condition effects',
        'under this architecture and fixed budget. They do not identify true camera',
        'pose,semantic scene content or general impossibility of neural forecasting.',
        'Past alignment without restoration may discard motion needed for native',
        'targets;hard future restoration may impose a wrong camera trajectory.',
        'Those mechanisms require their respective retained results,not assertion.', '',
        f'Past-only raw-actor known-easy +harm violations:{harms["past_only"]}/34.',
        f'Future-only raw-actor violations:{harms["future_only"]}/34.',
        'No raw-actor average is a full selected-policy safety pass. Original2%/',
        'support/NA rules remain;no future/test input or new confirmation access.',
        'A favorable secondary arm cannot silently replace the registered primary.',
        'No controller is trained and no deployment changes. One seed and exposed',
        'development sources remain insufficient for formal main or paper claims.']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure)+'\n')
    gates = ['# Camera-Operator Component Gates', '', '| Gate | Status |', '|---|---|',
        '| Full actual Torch treatment matrix | PASS:12models4000each |',
        '| Full identity-control scientific parity | PASS |', '| Real-data100+100vs200recovery | PASS |',
        '| Complete treatment refits | PASS:two fixed-first-design refits,not twelve |',
        '| Twelve new predictions and all scalars replay | PASS |',
        '| Cached both/baseline readout exact | PASS |',
        '| Fixed primary component screen | '+('PASS,exploratory' if passed else 'FAIL')+' |',
        '| New full selected-policy risk | NOT RUN |', '| Independent confirmation/three forecaster seeds | NOT RUN |',
        '| Deployment/submission readiness | FALSE |', '| Stage5C execution/SMC | FALSE |']
    (PUBLIC/'world_model_gate_camera_factor.md').write_text('\n'.join(gates)+'\n')
    lines = ['# 本轮相机操作拆分实验复现', '',
        '需要现有 native 数据、已校验过去图像变换代理和 pinned EqMotion 源码。',
        'arm64 `.venv-pytorch`，CPU4/inter-op1/workers0。不要重复启动活跃进程。', '', '```sh']
    for phase in ('admit', 'control --resume', 'train --resume', 'verify --resume', 'forecast', 'evaluate', 'replay'):
        lines.append('.venv-pytorch/bin/python scripts/run_m3w_moving_camera_factor.py '+phase)
    lines += ['.venv-pytorch/bin/python scripts/report_m3w_moving_camera_factor.py', '```', '',
        '已完成的 pilot/evaluate 收据不可覆盖；replay 重算12个新预测bank和全部指标，',
        '旧18个bank本轮只做hash与逐行ID校验，不说成重新推理。两个完整重训核验',
        '仅覆盖每个新arm固定第一design，不是12个模型全部完整重训。',
        '中断从同定义checkpoint恢复，不减少4000步、不改2%/support/NA规则。',
        '禁止 future endpoint、未来真实相机位姿、central velocity 和test endpoint goals。',
        'loss字段 mean_past_normalized_ADE 沿用旧命名，实际是未加权batch',
        'available-label ADE的packed图像坐标数值，不是米或统一跨域尺度。',
        '这是已有暴露A侧的模型组件实验，不是新独立测试或部署认证。']
    (PUBLIC/'operation_zh.md').write_text('\n'.join(lines)+'\n')
    figures(a, fits)
    print('Completed operator matrix reported;no deployment or submission claim.')


if __name__ == '__main__': main()
