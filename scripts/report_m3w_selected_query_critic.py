"""Export real training telemetry and separately verified critic policy evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_selected_query_critic as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups


def training():
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text()); fits = []
    for ref in freeze['fits']:
        path = ROOT/ref['path']
        if run.sha(path) != ref['sha256']: raise ValueError('Fixed training receipt changed')
        fits.append(json.loads(path.read_text()))
    reg = json.loads(run.REG.read_text())
    expected = {(c, seed, arm) for c in reg['contexts'] for seed in (17, 29, 43) for arm in run.api.ARMS}
    actual = {(r['context'], r['seed'], r['arm']) for r in fits}
    if (len(fits) != 36 or actual != expected or {r['step'] for r in fits} != {2000}
            or {r['parameters'] for r in fits} != {30726}):
        raise ValueError('All36 registered full fits required')
    return freeze, fits


def export_training():
    freeze, fits = training(); traces = []
    for r in fits:
        cp = r['checkpoint']; path = ROOT/cp['path']
        if run.sha(path) != cp['sha256']: raise ValueError('Full trained checkpoint changed')
        state = run.api.read_checkpoint(path)
        traces.append(dict(context=r['context'], seed=r['seed'], arm=r['arm'],
            checkpoint_sha256=cp['sha256'], trace=state['trace']))
    run.once(run.PUBLIC/'training_trace_metrics.json', dict(training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        fits=traces, raw_data_or_weights_included=False, original_training_trace=True,
        no_new_training_or_policy_readout=True))
    text, stats = training_text(freeze, fits, traces)
    (run.PUBLIC/'training_report.md').write_text(text)
    run.write(run.PUBLIC/'training_diagnostics.json', stats)
    training_plot(traces)
    print(json.dumps(dict(full_fits=36, primary_updates=72000, full_refit_updates=4000, policy_outcomes_read=False)))


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else None


def training_text(freeze, fits, traces):
    stats = {}
    lines = ['# Real selected-query critic training', '', '## Material Passport', '',
        'fresh_run:36 full native-arm64 Torch risk fits,72,000 primary gradient updates.',
        'Both first context/seed17 heads have separate2,000-step exact refits (4,000 additional updates).',
        'cached_verified: original forecast/cost heads, causal features, source inputs and selected-action identities.',
        'No newly trained trajectory forecaster, threshold selection or independent outcome readout.',
        f"Training PID{freeze['pid']} takes{freeze['seconds']:.3f}s,peak RSS{freeze['peak_RSS_bytes']}B.",
        'All36 heads have30,726 trainable parameters. Row-local and selected-set pairs share initialization,',
        'source-balanced draws, label masks, fitting-only normalizers, architecture size and loss weights.',
        'Benefit,reference and easy-reference predictions are frozen; only harm/easy-harm are corrected.', '',
        '## Fixed fitting-monitor loss', '',
        'These are16 fixed source-balanced training queries per head,not validation/test loss or full-dataset means.',
        'Every candidate is final-step2,000; no checkpoint is selected from these curves.', '',
        '|Arm|Heads|Monitor total decreased|Median initial total|Median final total|Median initial query score|Median final query score|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        subset = [r for r in fits if r['arm'] == arm]
        initial = [r['first_trace']['monitor'] for r in subset]; final = [r['final_trace']['monitor'] for r in subset]
        snapshots = [row for t in traces if t['arm'] == arm for row in t['trace'] if 'pointwise_gradient_norm' in row]
        ratios = [r['query_over_pointwise_gradient'] for r in snapshots if r['query_over_pointwise_gradient'] is not None]
        cosines = [r['query_pointwise_cosine'] for r in snapshots if r['query_pointwise_cosine'] is not None]
        stats[arm] = dict(heads=len(subset),
            monitor_total_decreased=sum(b['total'] < a['total'] for a, b in zip(initial, final)),
            median_initial_total=float(np.median([r['total'] for r in initial])),
            median_final_total=float(np.median([r['total'] for r in final])),
            median_initial_query_score=float(np.median([r['selected_query_score'] for r in initial])),
            median_final_query_score=float(np.median([r['selected_query_score'] for r in final])),
            gradient_snapshots=len(snapshots), query_pointwise_norm_ratio_quantiles=quantiles(ratios),
            query_pointwise_cosine_quantiles=quantiles(cosines),
            zero_query_gradient_snapshots=sum(r['selected_query_gradient_norm'] == 0 for r in snapshots),
            undefined_pointwise_denominator_snapshots=sum(r['query_over_pointwise_gradient'] is None for r in snapshots),
            negative_gradient_cosine_snapshots=sum(v < 0 for v in cosines),
            query_gradient_below_one_percent_pointwise_snapshots=sum(v < .01 for v in ratios),
            unknown_rows_drawn=sum(r['unknown_rows_drawn'] for r in subset), unknown_rows_with_loss=0)
        s = stats[arm]
        lines.append(f"|{arm}|{len(subset)}|{s['monitor_total_decreased']}|{s['median_initial_total']:.9f}|"
            f"{s['median_final_total']:.9f}|{s['median_initial_query_score']:.9f}|{s['median_final_query_score']:.9f}|")
    lines += ['', '## Actual component-gradient telemetry', '',
        'Norm ratios compare the two unweighted component gradients from the SAME actual update batch.',
        'Their registered weights are both0.5,so the ratio also applies after these weights.',
        'This is not a ratio between medians drawn from different experiments.', '',
        '|Arm|Snapshots|Query/point norm min,Q1,median,Q3,max|Zero query gradient|Negative cosine|Query gradient below1%|',
        '|---|---:|---|---:|---:|---:|']
    for a, s in stats.items():
        lines.append(f"|{a}|{s['gradient_snapshots']}|{s['query_pointwise_norm_ratio_quantiles']}|"
            f"{s['zero_query_gradient_snapshots']}|{s['negative_gradient_cosine_snapshots']}|"
            f"{s['query_gradient_below_one_percent_pointwise_snapshots']}|")
    lines += ['', 'All requested candidate actions and current-query peers are retained before supervision masks.',
        'Unknown labels contribute no supervised loss. The query target is supported only for nonempty,',
        'fully known original selected subsets; empty/unknown subsets are not zero-risk targets.',
        'Lower fitting loss is optimization evidence,not calibrated post-selection risk or independent lift.',
        'Gradient weakness/conflict is measured but does not by itself identify the unique cause of policy failure.',
        'The new policy still reassigns actions after training on the original selected subset,which can create',
        'a final-selection mismatch. Full recording risk must be evaluated separately.', '',
        '## Scope', '',
        'All source roles are exposed fitting/resubstitution. Three critic seeds are not three forecast producers.',
        'Task:obs8/pred12 at raw stride12, detector-silver/local image coordinates.',
        'Not metric,seconds-level,human gold,true3D,foundation or newly successful world dynamics.',
        'No deployment change,Stage5C execution,SMC or submission-ready claim.', '']
    return '\n'.join(lines), dict(arms=stats, primary_optimizer_updates=72000,
        full_refit_updates=4000, fitting_monitor_not_validation=True, new_trajectory_forecaster_fits=0,
        independent_confirmation=False, deployment_changed=False, submission_ready=False)


def training_plot(traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for arm, color in zip(run.api.ARMS, ('#666666', '#007C91')):
        rows = [t['trace'] for t in traces if t['arm'] == arm]
        steps = [r['step'] for r in rows[0]]
        if any([r['step'] for r in rr] != steps for rr in rows): raise ValueError('Complete same training steps required')
        value = np.median([[r['monitor']['total'] for r in rr] for rr in rows], axis=0)
        axes[0].plot(steps, value, label=arm, color=color)
        snapshots = {}
        for rr in rows:
            for r in rr:
                value = r.get('query_over_pointwise_gradient')
                if value is not None and value > 0: snapshots.setdefault(r['step'], []).append(value)
        axes[1].plot(sorted(snapshots), [np.median(snapshots[k]) for k in sorted(snapshots)], label=arm, color=color)
    axes[0].set_title('Median fixed fitting-monitor loss', fontsize=11)
    axes[0].set_ylabel('Normalized objective (not validation error)')
    axes[1].set_title('Same-batch query / point gradient norm', fontsize=11)
    axes[1].set_ylabel('Median positive ratio; zero counts in report'); axes[1].set_yscale('log')
    for ax in axes: ax.set_xlabel('Actual training update'); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); folder = run.PUBLIC/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'critic_training.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'critic_training.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def build():
    complete, summary, verification = [json.loads((run.PUBLIC/n).read_text()) for n in
        ('complete.json', 'summary.json', 'verification.json')]
    if (not verification['all72groups_exact'] or not verification['all144parent_full_policy_controls_exact']
            or verification['summary_sha256'] != run.sha(run.PUBLIC/'summary.json')
            or verification['complete_sha256'] != run.sha(run.PUBLIC/'complete.json')
            or complete['summary_sha256'] != run.sha(run.PUBLIC/'summary.json')):
        raise ValueError('Complete separately replayed critic readout required')
    reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Registered critic dependency changed: '+rel)
    freeze, fits = training()
    if complete['training_freeze_sha256'] != run.sha(run.PUBLIC/'training_freeze.json'):
        raise ValueError('Readout must bind original complete training freeze')
    grouped = aggregate_groups(run.NAME)
    rows = [grouped[Path(ref['path']).name] for ref in complete['groups']]
    if run.evaluation.summarize(rows, reg['contexts'], json.loads(run.CONFIG.read_text())) != summary:
        raise ValueError('Canonical complete critic reduction differs')
    reasons = {a: Counter() for a in run.decoder.NAMES[2:]}; nonoptimal = Counter(); failures = Counter()
    for r in rows:
        for a in reasons: reasons[a].update(r['result']['allocation']['reason_counts'][a])
        nonoptimal.update(r['result']['allocation']['nonoptimal_queries'])
        for record in r['result']['recordings'].values():
            failures.update(k for k, v in record['set_joint']['budget_status']['conditions'].items() if not v)
    replay = run.PUBLIC/'replay_runtime.json'
    return dict(complete=complete, summary=summary, rows=rows, training_freeze=freeze, fits=fits,
        training_diagnostics=json.loads((run.PUBLIC/'training_diagnostics.json').read_text()),
        reason_counts={a: dict(v) for a, v in reasons.items()}, nonoptimal_queries=dict(nonoptimal),
        primary_failure_conditions=dict(failures),
        replay_runtime=json.loads((replay if replay.exists() else run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(value):
    return f"{value['point']:+.6f} [{value['CI95'][0]:+.6f}, {value['CI95'][1]:+.6f}]" if value['CI95'] else 'not estimable'


def render(value):
    s, c = value['summary'], value['complete']
    lines = ['# Selected-query risk critic: complete paired readout', '', '## Material Passport', '',
        'fresh_run:36 matched real Torch risk fits,72,000 primary updates,4,000 exact-refit updates,full inference/allocation/scoring.',
        'cached_verified: frozen upstream forecasts,cost heads,causal features,original selected sets and source inputs.',
        'This is exposed SOURCE fitting/resubstitution,not independent confirmation or a new trajectory dynamics model.',
        f"Readout PID{c['pid']}: {c['seconds']:.3f}s;separate replay PID{value['replay_runtime']['pid']}: {value['replay_runtime']['seconds']:.3f}s.",
        'All72 groups and144 old full-policy controls match. No outcome-based checkpoint or threshold selection.', '',
        '## Controlled method', '',
        'The two30,726-parameter heads differ only in row-local versus original-selected-set hidden context.',
        'Both receive count metadata and learn the same all-action/selected-query objective with the same fitting-only RMS.',
        'All seven candidates are rescored. Benefit/reference/easy-reference are exactly frozen;harm/easy-harm change.',
        'The same joint allocator uses each head\'s independent per-query count;extra matched controls use their common minimum.',
        'Labels and eventual support masks do not enter inference/context. Unknown rows remain in the cohort and context.', '',
        '## Full-policy utility and unchanged risk', '',
        'Known ADE gain versus the frozen floor (%);conditional95% locality bootstrap,3,000 draws over12 exposed localities.',
        'Repeated context/critic-seed views are averaged first. There is one fixed forecaster seed,not three producers.', '',
        '|Policy|Known ADE gain [95% CI]|Completion-lower floor gain pp [95% CI]|Supported record views|Known easy violations|Undefined easy|Selected views|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        r = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{fmt(s['floor_completion_lower_pp'][a])}|"
            f"{r['supported_record_views']}/{r['required_record_views']}|{r['known_easy_risk_violations']}|"
            f"{r['undefined_easy_record_views']}|{r['selected']}|")
    lines += ['', '|Prespecified contrast|Full known increment pp [95% CI]|Full wholly-unknown completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for name, v in s['comparisons'].items(): lines.append(f"|{name}|{fmt(v['full_known_pp'])}|{fmt(v['full_lower_pp'])}|")
    lines += ['', 'The context-versus-local full-policy contrast is primary. The explicit matched control is utility-only.',
        'Generic empty count-trimmed contrasts remain not estimable in the aggregate archive;no zero-risk certificate is invented.',
        'A context-parent improvement alone cannot isolate pooling:both critics also share new supervision,normalization and architecture.', '',
        '## Risk learning and actual training', '',
        '|Cost estimator|All known legal candidate risk MSE [95% CI]|', '|---|---:|']
    for name, v in s['calibration_risk_MSE'].items(): lines.append(f"|{name}|{fmt(v)}|")
    lines += ['', 'MSE uses fitting-only causal scales and risk RMS. It is not ECE,a calibrated probability or observed safety.',
        '[Actual training curves and gradient telemetry](training_report.md) are separate from policy efficacy.',
        f"Original recording failure conditions for selected-set joint: {value['primary_failure_conditions']}.",
        f"Nonoptimal query views: {value['nonoptimal_queries']}.",
        'Feasible anchors are retained on solver rejection. Original all/easy2% and easy-degradation upper criteria remain.', '',
        '## Fitting gates', '']
    lines += [f"- {k}: {'pass' if v else 'fail'}" for k, v in s['fitting_gates'].items()]
    lines += ['', f"{sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])} fitting checks;advancement {s['fitting_advancement_gate']}.",
        'No deployment or independent-evaluation promotion follows merely from loss reduction or fewer interventions.', '',
        '## Limits', '',
        'The same149,510 unique agent-target rows and18,766 current-frame queries across74 recordings/12 localities',
        'generate reused views. Full-roster FDE is not estimable if a required endpoint is unavailable.',
        'ADE uses available future labels;wholly-unknown completion bounds do not certify missing interior labels.',
        'The query target concerns the original selected set;the recalibrated decoder may select a different set.',
        'Predicted budget feasibility does not prove calibration under that final selection or finite-recording compliance.',
        'Pair geometry is a predicted-path proxy among target queries,not verified physical safety or all-visible-agent modeling.',
        'New physical-only reoptimization with these critics is not run;the preceding bank-control result cannot be relabelled as such.',
        'This is obs8/pred12 at raw stride12,not historical raw-frame t+50. No metric,seconds,human-gold,true3D or foundation claim.',
        'Independent roles,Stage5C execution and SMC remain closed;submission readiness remains false.', '']
    return '\n'.join(lines)


def complete_report():
    value = build(); s = value['summary']; p = run.PUBLIC
    run.once(p/'replay_runtime.json', value['replay_runtime'])
    (p/'readout.md').write_text(render(value))
    run.write(p/'mechanism_diagnostics.json', {k: value[k] for k in ('reason_counts', 'nonoptimal_queries', 'primary_failure_conditions')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    names = ('parent_independent', 'parent_joint', 'row_joint', 'set_joint', 'row_matched', 'set_matched')
    labels = ('Original independent', 'Original joint', 'Local risk + joint', 'Set risk + joint', 'Local common count', 'Set common count')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for i, a in enumerate(names):
        v = s['accuracy'][a]; low, high = v['CI95']
        axes[0].errorbar(v['point'], i, xerr=[[max(0, v['point']-low)], [max(0, high-v['point'])]],
            fmt='o', capsize=3, color='#007C91' if a.startswith('set_') else '#666666')
        axes[1].barh(i-.14, s['safety'][a]['supported_record_views'], height=.27, color='#007C91')
        axes[1].barh(i+.14, s['safety'][a]['known_easy_risk_violations'], height=.27, color='#B44D48')
    axes[0].set_yticks(range(len(names)), labels); axes[0].invert_yaxis()
    axes[0].set_xlabel('Known ADE gain over floor (%)'); axes[0].axvline(0, color='#888888', linewidth=.8)
    axes[0].grid(axis='x', alpha=.2); axes[1].set_yticks(range(len(names)), labels); axes[1].invert_yaxis()
    axes[1].set_xlim(0, 450); axes[1].set_xlabel('Recording/context/seed views (444 required)')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color='#007C91', label='Supported'), Patch(color='#B44D48', label='Known easy violation')],
        frameon=False, fontsize=8)
    fig.suptitle('Exposed fitting evidence;mean accuracy is not complete-policy safety', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); path = p/'figures/critic_policy.svg'
    fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'critic_policy.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), training_freeze_sha256=run.sha(p/'training_freeze.json'),
        training_trace_sha256=run.sha(p/'training_trace_metrics.json'), report_code_sha256=run.sha(Path(__file__)),
        report_sha256=run.sha(p/'readout.md'), figure_sha256=run.sha(path),
        training_figure_sha256=run.sha(p/'figures/critic_training.svg'), scientific_rescoring=False,
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, fitting_gates=s['fitting_gates'], deployment_changed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('training', 'complete'))
    args = parser.parse_args(); export_training() if args.phase == 'training' else complete_report()
