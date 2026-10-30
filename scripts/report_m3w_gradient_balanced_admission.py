"""Render the fixed norm-balancing result and actual optimization telemetry."""
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_gradient_balanced_admission as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from scripts.run_m3w_selected_action_attribution import pool


def build():
    public = run.PUBLIC
    complete, summary, verification, freeze = [json.loads((public / name).read_text()) for name in
        ('complete.json', 'summary.json', 'verification.json', 'training_freeze.json')]
    if (verification['summary_sha256'] != run.sha(public / 'summary.json')
            or verification['complete_sha256'] != run.sha(public / 'complete.json')
            or not verification['all72groups_exact']
            or not verification['all144cached_control_predictions_actions_full_scores_exact']
            or freeze['count'] != 18 or len(freeze['cached_verified_controls']) != 36
            or complete['summary_sha256'] != run.sha(public / 'summary.json')):
        raise ValueError('Complete exact verified readout required')
    for reg in (run.REG, public / 'readout_registration.json', run.monitor.REG):
        for rel, digest in json.loads(reg.read_text())['bindings'].items():
            if run.sha(ROOT / rel) != digest:
                raise ValueError('Scientific dependency changed: ' + rel)
    models = run.models()
    aggregate = aggregate_groups(run.NAME)
    # Preserve the original fixed accumulation order when reconstructing floats.
    rows = [aggregate[Path(ref['path']).name] for ref in complete['groups']]
    cfg = json.loads(run.CONFIG.read_text())
    from src.evaluation import m3w_gradient_balance_readout as evaluation
    if evaluation.summarize(rows, json.loads(run.REG.read_text())['contexts'], cfg, freeze) != summary:
        raise ValueError('Complete summary reconstruction differs')
    first = sorted(json.loads(run.REG.read_text())['contexts'])[0]
    refit = json.loads((public / 'first_full_refit.json').read_text())
    if (not refit['scientific_state_exact'] or run.sha(run.checkpoint(first, 17)) != refit['checkpoint_sha256']
            or run.sha(run.checkpoint(first, 17, 'full_refit')) != refit['refit_checkpoint_sha256']):
        raise ValueError('Full refit identity differs')
    traces, telemetry = {}, []
    for key, row in sorted(models.items()):
        if key[2] != 'balanced_score':
            continue
        state = run.parent.parent.frozen.api.completion.base.read_checkpoint(ROOT / row['checkpoint']['path'])
        if state['step'] != 2000 or state['unknown_rows_with_loss'] or state['parameters'] != 30405:
            raise ValueError('Full same-size real Torch training required')
        run.parent.parent.frozen.api.completion.base.exact(state['gradient_balance'], row['gradient_balance'])
        traces['|'.join(map(str, key))] = state['trace']
        telemetry.append(state['gradient_balance'])
    snapshots = [r for s in telemetry for r in s['rows']]
    nonzero = [r['weighted_auxiliary_over_anchor'] for r in snapshots if r['scale'] > 0]
    pressure = dict(total_steps=sum(v['step'] for v in telemetry),
        zero_scale_steps=sum(v['zero_scale_steps'] for v in telemetry),
        capped_steps=sum(v['capped_steps'] for v in telemetry), snapshots=len(snapshots),
        nonzero_snapshots=len(nonzero), nonzero_weighted_ratio_quantiles=np.quantile(nonzero, [0, .1, .5, .9, 1]).tolist() if nonzero else None,
        maximum_scale=max(v['scale_max'] for v in telemetry), rule=run.api.RULE)
    failures = Counter()
    for row in rows:
        for record in row['result']['recordings'].values():
            v = record['balanced_score']
            failures.update('condition_failed_'+k for k, passed in v['budget_status']['conditions'].items() if not passed)
            for label, field in (('all', 'known_positive_risk'), ('easy', 'known_easy_positive_risk')):
                upper, known = v[label+'_selected_risk_upper'], v[field]
                if upper is None:
                    failures[label+'_undefined'] += 1
                elif upper > .02+1e-12 and known is not None and known <= .02+1e-12:
                    failures[label+'_completion_upper_fails_despite_known_budget'] += 1
    oracle_path = run.parent.parent.frozen.PUBLIC / 'summary.json'
    oracle_check = json.loads((run.parent.parent.frozen.PUBLIC / 'verification.json').read_text())
    if oracle_check['summary_sha256'] != run.sha(oracle_path):
        raise ValueError('Cached bank opportunity evidence changed')
    oracle = json.loads(oracle_path.read_text())['bank_opportunity_diagnostic']['new_neural']
    return dict(summary=summary, complete=complete, training=freeze, models=models, groups=rows,
        pressure=pressure, failure_conditions=dict(failures),
        cached_oracle=dict(oracle, result_source='cached_verified_label_aware_diagnostic_not_inference',
            source_sha256=run.sha(oracle_path)), traces=traces,
        attribution={o: pool([r['result']['attribution'][o] for r in rows])
            for o in evaluation.OBJECTIVES},
        verification_runtime=json.loads((run.PRIVATE / 'verification_runtime.json').read_text()),
        new_checkpoint_bytes=sum(r['checkpoint']['bytes'] for k, r in models.items() if k[2] == 'balanced_score'))


def fmt(v):
    if v['point'] is None:
        return 'not estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def render(value):
    s, t, c, p = value['summary'], value['training'], value['complete'], value['pressure']
    names = ('upper_score', 'admitted_score', 'balanced_score')
    lines = ['# Capped norm-parity admission learning', '', '## Material Passport', '',
        'Fresh run: 18 newly trained cost heads and fresh inference. Cached verified: 36 frozen upper/unit control models.',
        'All data are exposed SOURCE fitting/resubstitution. Independent confirmation, deployment and submission readiness remain false.', '',
        '## Intervention and actual execution', '',
        'The preceding exact training replay found frequent auxiliary activation but a median',
        'parameter-gradient ratio of only 0.003632%. This trial changes only its detached scale:',
        '`lambda=min(1e6, ||grad anchor||/||grad auxiliary||)` when defined; otherwise zero.',
        'The same five-moment/ranking/upper anchor, auxiliary, model, features, normalization,',
        'sample draws, candidate bank and 2% inference rule remain. This is not full GradNorm,',
        'a new deferral framework, moment calibration or a safety theorem.', '',
        f"All 18 fresh heads complete 2,000 steps: 36,000 primary updates, 2,000 exact-refit updates and 400 pilot updates.",
        f"Training PID {t['pid']} takes {t['seconds']:.3f}s, peak RSS {t['peak_RSS_bytes']}B.",
        'All 36 frozen controls match common initialization/preprocessing/RNG/draw design; they are not claimed as newly trained.',
        f"Readout PID {c['pid']} takes {c['seconds']:.3f}s. A separate process repeats all 72 groups and the summary exactly.",
        'Both controls reproduce all 144 prediction/action/full-policy metric checks.', '',
        '## Did the optimization mechanism operate?', '',
        f"Across {p['total_steps']} fresh steps: {p['zero_scale_steps']} zero-scale and {p['capped_steps']} capped steps.",
        f"At {p['nonzero_snapshots']}/{p['snapshots']} monitored snapshots the scale is positive.",
        f"Nonzero achieved auxiliary/anchor gradient-ratio quantiles (min,p10,median,p90,max): {p['nonzero_weighted_ratio_quantiles']}.",
        'These are pre-clipping gradient ratios, not AdamW update ratios, calibrated risks or independent samples.',
        'The numerical cap is retained and disclosed, not adjusted after scores.', '',
        '## Complete-policy accuracy and risk', '',
        'Known ADE gain (%) versus the frozen floor, equal locality with 95% bootstrap intervals.',
        'Three controller seeds and 3,000 locality resamples after context/seed averaging.', '',
        '|Objective|Known ADE gain|Supported recording views|Known easy violations|Known all violations|Undefined easy views|',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        r = s['safety'][name]
        lines.append(f"|{name}|{fmt(s['accuracy'][name])}|{r['supported_record_views']}/{r['required_record_views']}|"
            f"{r['known_easy_risk_violations']}|{r['known_all_risk_violations']}|{r['undefined_easy_record_views']}|")
    lines += ['', '|Fixed contrast|Known pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|', '|---|---:|---:|']
    for name, data in s['comparisons'].items():
        lines.append(f"|{name}|{fmt(data['full_known_pp'])}|{fmt(data['full_lower_pp'])}|")
    lines.append(f"|Same balanced head actual neural use|{fmt(s['actual_neural_use']['full_known_pp'])}|{fmt(s['actual_neural_use']['full_lower_pp'])}|")
    lines += ['', 'A positive secondary or known-only mean cannot replace complete-policy risk or the fixed primary.',
        'Same-count controls are utility-only; empty is not estimable. The complete-roster FDE is',
        'not estimable when a required view lacks endpoints. Completion sensitivity bounds wholly',
        'unknown rows only, not all missing positions inside partially observed ADE labels.', '',
        '## Selected easy-harm diagnosis', '',
        '|Objective|Selected known|Selected unknown|Easy false admissions|Observed easy harm|Predicted easy harm|',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        a = value['attribution'][name]; ch = a['channels']['easy']
        lines.append(f"|{name}|{a['selected_known_rows']}|{a['selected_unknown_rows']}|{ch['false_admitted_known_rows']}|"
            f"{ch['observed_harm_mass']:.6f}|{ch['predicted_harm_mass']:.6f}|")
    lines += ['', 'Arithmetic witnesses use evaluation labels only and are not inference-time policies or causal identification.',
        'Counts/masses are overlapping fitting views, not independent new data. The same 149,510 unique',
        'queries, 74 recordings and 12 localities remain; unknown outcomes are not imputed.', '',
        '## Remaining risk and support failures', '',
        f"Failure counts (overlapping, not a partition): {value['failure_conditions']}.",
        'The same frozen bank has cached label-aware positive feasible opportunity in all 148',
        'seed-deduplicated context/record views. This uses evaluation labels and whole-record',
        'allocation; it does not show inference-time identifiability or simultaneous-frame feasibility.',
        'It is cached verified evidence, not a fresh oracle run or a learned policy result.', '', '## Fixed fitting gates', '']
    lines += [f"- {key}: {'pass' if passed else 'fail'}" for key, passed in s['fitting_gates'].items()]
    lines += ['', f"Progress gates: {sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])}; advancement: {s['fitting_advancement_gate']}.",
        'Even passing fitting gates would not establish independent generalization or submission readiness.', '',
        '## Scope and reproduction', '',
        'Native arm64 CPU4/interop1/workers0; checkpoint100/heartbeat50, explicit resume and bounded runtime.',
        'All 18 actual loss traces and balance statistics are retained; raw inputs, caches and weights remain private.',
        'This is supplied detector-silver top-down pixel/raw-frame forecasting, with eight past and twelve requested',
        'steps at raw stride12. It is not the supplemental raw-frame t+50 task, metric, seconds-level, human gold,',
        'true3D or a foundation model. No Stage5C execution or SMC. No new scene-joint dynamics claim.', '']
    return '\n'.join(lines)


def main():
    value = build(); s = value['summary']
    (run.PUBLIC / 'readout.md').write_text(render(value))
    run.write(run.PUBLIC / 'optimization_pressure.json', value['pressure'])
    run.write(run.PUBLIC / 'failure_conditions.json', dict(counts=value['failure_conditions'],
        overlapping_not_partition=True, cached_bank_opportunity=value['cached_oracle']))
    run.write(run.PUBLIC / 'training_traces.json', dict(fresh_heads=18, actual_optimizer_traces=value['traces']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1.5]})
    contrasts = (s['comparisons']['balanced_minus_admitted']['full_known_pp'],
                 s['comparisons']['balanced_minus_upper']['full_known_pp'], s['actual_neural_use']['full_known_pp'])
    for i, (v, color) in enumerate(zip(contrasts, ('#B44D48', '#555555', '#007C91'))):
        low, high = v['CI95']
        ax[0].errorbar(v['point'], i, xerr=[[v['point']-low], [high-v['point']]], fmt='o', capsize=4, color=color)
    ax[0].set_yticks(range(3), ['Balanced - unit', 'Balanced - upper', 'Actual neural use'])
    ax[0].invert_yaxis(); ax[0].axvline(0, color='#888888', linewidth=.8)
    ax[0].set_xlabel('Paired known ADE increment (pp)'); ax[0].grid(axis='x', alpha=.2)
    ax[0].set_title('Fitting utility: locality bootstrap 95% CI', fontsize=11)
    names = ('upper_score', 'admitted_score', 'balanced_score'); x = np.arange(3)
    for j, (key, label, color) in enumerate(zip(
        ('supported_record_views', 'known_easy_risk_violations', 'undefined_easy_record_views'),
        ('Supported', 'Known easy violation', 'Undefined easy'), ('#007C91', '#B44D48', '#D4A43A'))):
        ax[1].bar(x+(j-1)*.24, [s['safety'][o][key] for o in names], width=.23, label=label, color=color)
    ax[1].set_xticks(x, ['Upper', 'Unit admission', 'Norm balanced'])
    ax[1].set_ylim(0, 480); ax[1].set_ylabel('Recording/context/seed views (444 required)')
    ax[1].legend(frameon=False, fontsize=9, loc='upper right'); ax[1].set_title('Original complete-policy risk contract', fontsize=11)
    fig.suptitle('Exposed SOURCE fitting only; no independent or deployment promotion', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    folder = run.PUBLIC / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'balanced_summary.svg', bbox_inches='tight', metadata={'Date': None})
    svg = folder / 'balanced_summary.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE / 'balanced_summary.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    run.write(run.PUBLIC / 'evidence.json', dict(summary_sha256=run.sha(run.PUBLIC / 'summary.json'),
        complete_sha256=run.sha(run.PUBLIC / 'complete.json'), verification_sha256=run.sha(run.PUBLIC / 'verification.json'),
        report_code_sha256=run.sha(Path(__file__)), readout_sha256=run.sha(run.PUBLIC / 'readout.md'),
        traces_sha256=run.sha(run.PUBLIC / 'training_traces.json'), pressure_sha256=run.sha(run.PUBLIC / 'optimization_pressure.json'),
        failure_conditions_sha256=run.sha(run.PUBLIC / 'failure_conditions.json'),
        figure_sha256=run.sha(folder / 'balanced_summary.svg'), all18fresh36cached_weights_verified=True,
        scientific_rescoring=False, independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, gates=s['fitting_gates'], pressure=value['pressure'])))


if __name__ == '__main__':
    main()
