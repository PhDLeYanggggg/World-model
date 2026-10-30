"""Report real matched fits in common loss units and separately replayed policies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_query_unit_critic as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import pack, verify

np = run.np


def common_curve(trace, mode):
    if mode not in ('old', 'new'): raise ValueError('Explicit original/new loss units required')
    steps = [r['step'] for r in trace]
    if not steps or steps != sorted(set(steps)): raise ValueError('Unique ordered trace steps required')
    key = 'selected_query_score_original_units' if mode == 'new' else 'selected_query_score'
    if any(key not in r['monitor'] for r in trace): raise ValueError('Original-unit query telemetry required')
    point = [r['monitor']['pointwise_risk'] for r in trace]
    query = [r['monitor'][key] for r in trace]
    return dict(steps=steps, original_objective=[.5*(p+q) for p, q in zip(point, query)], pointwise=point, query=query)


def training():
    p = run.PUBLIC; reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Registered one-factor design changed: '+rel)
    freeze = json.loads((p/'training_freeze.json').read_text()); traces = []; fits = []
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('New fit receipt changed')
        row = json.loads((ROOT/ref['path']).read_text()); states = {}
        for mode, key in (('new', 'checkpoint'), ('old', 'cached_control')):
            cp = row[key]
            if run.sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Matched checkpoint changed')
            states[mode] = run.api.parent.read_checkpoint(ROOT/cp['path'])
            if states[mode]['step'] != 2000 or states[mode]['parameters'] != 30726:
                raise ValueError('Both arms need the same complete original budget')
        run.api.parent.assert_matched(states['new'], states['old'])
        if row['query_normalizer'] != states['new']['identity']['query_unit_normalizer']:
            raise ValueError('Fitting-only normalizer identity differs')
        traces.append(dict(context=row['context'], seed=row['seed'], arm=row['arm'],
            query_normalizer=row['query_normalizer'], old=states['old']['trace'], new=states['new']['trace'],
            unknown_rows_drawn=states['new']['unknown_rows_drawn'],
            unknown_rows_with_loss=states['new']['unknown_rows_with_loss']))
        fits.append(row)
    expected = {(c, seed, arm) for c in reg['contexts'] for seed in (17, 29, 43) for arm in run.api.parent.ARMS}
    if len(fits) != 36 or {(r['context'], r['seed'], r['arm']) for r in fits} != expected:
        raise ValueError('All36 original paired fits required')
    for arm in run.api.parent.ARMS:
        if not json.loads((p/('first_full_refit_'+arm+'.json')).read_text())['scientific_state_exact']:
            raise ValueError('Both separate full refits must agree')
    return freeze, fits, traces


def training_stats(traces):
    result = {}
    for arm in run.api.parent.ARMS:
        pairs = [r for r in traces if r['arm'] == arm]; table = {}
        for mode in ('old', 'new'):
            curves = [common_curve(r[mode], mode) for r in pairs]
            if any(c['steps'] != curves[0]['steps'] for c in curves): raise ValueError('Same full trace steps required')
            snapshots = [t for r in pairs for t in r[mode] if 'pointwise_gradient_norm' in t]
            ratios = [t['query_over_pointwise_gradient'] for t in snapshots if t['query_over_pointwise_gradient'] is not None]
            table[mode] = dict(heads=len(pairs),
                original_unit_query_decreased_heads=sum(c['query'][-1] < c['query'][0] for c in curves),
                median_initial_original_unit_query=float(np.median([c['query'][0] for c in curves])),
                median_final_original_unit_query=float(np.median([c['query'][-1] for c in curves])),
                median_final_pointwise=float(np.median([c['pointwise'][-1] for c in curves])),
                gradient_snapshots=len(snapshots),
                query_pointwise_gradient_ratio_quantiles=np.quantile(ratios, [0, .25, .5, .75, 1]).tolist() if ratios else None,
                zero_query_gradient_snapshots=sum(t['selected_query_gradient_norm'] == 0 for t in snapshots),
                negative_gradient_cosine_snapshots=sum(t['query_pointwise_cosine'] is not None and t['query_pointwise_cosine'] < 0 for t in snapshots))
        result[arm] = dict(modes=table, unknown_rows_drawn=sum(r['unknown_rows_drawn'] for r in pairs),
            unknown_rows_with_loss=sum(r['unknown_rows_with_loss'] for r in pairs))
    return dict(arms=result, fresh_primary_updates=72000, exact_refit_updates=4000,
        cached_verified_control_heads=36, common_unit_comparison=True, monitor_queries_per_head=16,
        fitting_monitor_not_validation=True, independent_confirmation=False, deployment_changed=False)


def training_text(freeze, stats):
    lines = ['# One-factor query-unit repair: real training', '',
        'fresh_run:36 complete real Torch critics,72,000 updates plus4,000 separate exact-refit updates.',
        'cached_verified:36 original critics,unchanged causal feature/source inputs,forecasts and sampling design.',
        f"Training PID{freeze['pid']}: {freeze['seconds']:.3f}s,peak RSS{freeze['peak_RSS_bytes']}B.",
        'Native arm64 CPU4/interop1/workers0. Both full first-head refits agree exactly.',
        'Only the query-score denominator changes. All original random draws,initial states,architecture,',
        'pointwise loss,0.5/0.5 weights and allocation rules stay fixed. Unknown labels never supervise.', '',
        '## Common-unit fitting telemetry', '',
        'Each curve monitors16 fixed source-fitting queries,not validation/test or full-dataset means.',
        'New query scores below are converted to ORIGINAL units before comparison. New total loss is',
        'not numerically comparable to old total loss. Step2,000 is fixed;these curves select no checkpoint.', '',
        '|Arm|Loss units during training|Initial original-unit query score|Final original-unit query score|Heads decreased|Final point loss|Query/point gradient median|',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, pair in stats['arms'].items():
        for mode, v in pair['modes'].items():
            q = v['query_pointwise_gradient_ratio_quantiles']
            lines.append(f"|{arm}|{mode}|{v['median_initial_original_unit_query']:.9f}|{v['median_final_original_unit_query']:.9f}|"
                f"{v['original_unit_query_decreased_heads']}/{v['heads']}|{v['median_final_pointwise']:.9f}|{q[2] if q else None}|")
    lines += ['', 'Gradient ratios compare unweighted terms on the same actual update batch;both weights are0.5.',
        'Increasing the query gradient is a manipulation check,not a safety certificate or unique causal explanation.',
        'Original selected-set fitting may still differ from the final reoptimized policy selection.',
        'Protected-output finite-label interpolation limits are reported separately and not relaxed.', '',
        '## Boundaries', '', 'Exposed SOURCE fitting/resubstitution only,not whole-pipeline OOF or independent confirmation.',
        'Task:obs8/pred12 at raw stride12;detector-silver/local-image coordinates.',
        'No new trajectory forecaster,metric,seconds-level,human-gold,true3D or foundation claim.',
        'No deployment change,Stage5C execution,SMC or submission-ready claim.', '']
    return '\n'.join(lines)


def training_plot(traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = ((axes[0, 0], 'original_objective', 'Objective in original units'),
        (axes[0, 1], 'query', 'Query error in original units'),
        (axes[1, 0], 'pointwise', 'Unchanged pointwise risk loss'))
    for arm, color in zip(run.api.parent.ARMS, ('#666666', '#007C91')):
        pairs = [r for r in traces if r['arm'] == arm]
        for mode, style in (('old', '--'), ('new', '-')):
            curves = [common_curve(r[mode], mode) for r in pairs]
            for ax, key, title in panels:
                ax.plot(curves[0]['steps'], np.median([c[key] for c in curves], 0),
                    style, color=color, label=arm+' / '+mode)
                ax.set_title(title); ax.set_ylabel('Median fixed fitting monitor')
            snapshots = {}
            for pair in pairs:
                for row in pair[mode]:
                    value = row.get('query_over_pointwise_gradient')
                    if value is not None and value > 0: snapshots.setdefault(row['step'], []).append(value)
            axes[1, 1].plot(sorted(snapshots), [np.median(snapshots[k]) for k in sorted(snapshots)],
                style, color=color, label=arm+' / '+mode)
    axes[1, 1].set_title('Same-batch query / point gradient'); axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel('Median positive norm ratio')
    for ax in axes.flat: ax.set_xlabel('Actual update'); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Training-unit manipulation, not validation or policy safety', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); folder = run.PUBLIC/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'query_unit_training.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'query_unit_training.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def export_training():
    freeze, _, traces = training(); p = run.PUBLIC; stats = training_stats(traces)
    run.once(p/'training_trace_metrics.json', dict(training_freeze_sha256=run.sha(p/'training_freeze.json'),
        traces=traces, raw_data_or_weights_included=False, common_unit_comparison=True))
    run.write(p/'training_diagnostics.json', stats); (p/'training_report.md').write_text(training_text(freeze, stats))
    training_plot(traces)
    print(json.dumps(dict(real_complete_fits=36, new_policy_outcomes_read=False, diagnostics=stats)))


def canonical(rows, reg, cfg):
    evaluation = run.parent.evaluation; bank = evaluation.bank
    summary = evaluation.summarize(rows, reg['contexts'], cfg)
    interval = lambda get: bank.legacy.interval(rows, get, cfg)
    contrasts = {arm: {mode+'_'+bound: interval(lambda r, a=arm, m=mode, b=bound: r['query_unit_contrasts'][a][m][b])
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')} for arm in ('row', 'set')}
    context_gates = summary['fitting_gates']
    gates = dict(complete36fresh_fits72groups=True,
        all144old_critic_controls_exact=all(r['old_critic_controls_exact'] for r in rows),
        original_matched_training=context_gates['paired_feature_initialization_and_training_draws'],
        same_count_context_controls=context_gates['same_count_controls_every_query'],
        new_units_set_minus_old_units_lower_CI_positive=contrasts['set']['full_lower_pp']['CI95'] is not None and contrasts['set']['full_lower_pp']['CI95'][0] > 0,
        positive_floor_lower_utility=context_gates['context_vs_floor_completion_lower_CI_positive'],
        complete_recording_risk=context_gates['complete_context_recording_risk'],
        main_optima_certified=context_gates['all_main_query_optima_certified'])
    summary.update(primary='query_unit_repair_set_joint_minus_old_units_set_joint',
        cached_verified_critic_controls=36, query_unit_contrasts=contrasts, context_secondary_gates=context_gates,
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        old_units_accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent'])
            for a in ('old_units_row_joint', 'old_units_set_joint')},
        old_units_safety={a: bank.safety(rows, a) for a in ('old_units_row_joint', 'old_units_set_joint')})
    return summary


def build():
    p = run.PUBLIC; complete, s, check = [json.loads((p/n).read_text()) for n in ('complete.json', 'summary.json', 'verification.json')]
    if (not check['all72groups_exact'] or not check['all144old_critic_controls_exact']
            or check['summary_sha256'] != run.sha(p/'summary.json')
            or check['complete_sha256'] != run.sha(p/'complete.json')
            or complete['training_freeze_sha256'] != run.sha(p/'training_freeze.json')):
        raise ValueError('Separate complete replay and unchanged trained heads required')
    reg = json.loads(run.REG.read_text()); cfg = json.loads(run.CONFIG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Frozen one-factor definition changed')
    groups = aggregate_groups(run.NAME); rows = [groups[Path(r['path']).name] for r in complete['groups']]
    if canonical(rows, reg, cfg) != s: raise ValueError('Canonical72-group reduction differs')
    nonoptimal = Counter(); reasons = Counter()
    for r in rows:
        nonoptimal.update(r['result']['allocation']['nonoptimal_queries'])
        reasons.update(r['result']['allocation']['reason_counts']['set_joint'])
    runtime = p/'replay_runtime.json'
    return dict(summary=s, complete=complete, rows=rows, nonoptimal_queries=dict(nonoptimal), reason_counts=dict(reasons),
        replay_runtime=json.loads((runtime if runtime.exists() else run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]" if v['CI95'] else 'not estimable'


def render(value):
    s = value['summary']; lines = ['# Query-unit repair: complete paired policy readout', '', '## Material Passport', '',
        'fresh_run:36 real Torch critics,72,000 primary updates,4,000 exact-refit updates,complete72-group inference/allocation/scoring.',
        'cached_verified:36 old critics and their immutable choices;old predictions freshly re-inferred and hash-checked,',
        'and all144 full old-policy scores exactly recomputed. Old choices are NOT called freshly reoptimized.',
        f"Readout PID{value['complete']['pid']}: {value['complete']['seconds']:.3f}s;separate replay PID{value['replay_runtime']['pid']}: {value['replay_runtime']['seconds']:.3f}s.",
        'Both complete runs agree on all72 groups and the canonical summary. No outcome-selected threshold or checkpoint.', '',
        '## Primary unit-repair comparison', '',
        'Conditional95% paired locality bootstrap,3,000 draws over12 exposed SOURCE localities.',
        'Context/critic-seed views are averaged first;three critic seeds are not three forecaster producers.', '',
        '|Arm, new units minus original units|Full known ADE increment pp [95% CI]|Full wholly-unknown completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for a in ('row', 'set'):
        v = s['query_unit_contrasts'][a]
        lines.append(f"|{a}|{fmt(v['full_known_pp'])}|{fmt(v['full_lower_pp'])}|")
    lines += ['', 'Selected-set full completion-lower increment is prespecified primary;row-local is a control.',
        'No uncomputed difference-of-differences interval or special context/normalization interaction claim.', '',
        '## Full policy utility and original risk', '',
        '|Policy|Known ADE gain over floor [95% CI]|Supported recording-risk views|Known easy violations|Undefined easy|Selected views|',
        '|---|---:|---:|---:|---:|---:|']
    for a in (*run.parent.evaluation.NAMES, 'old_units_row_joint', 'old_units_set_joint'):
        old = a.startswith('old_units_'); r = (s['old_units_safety'] if old else s['safety'])[a]
        v = (s['old_units_accuracy'] if old else s['accuracy'])[a]
        lines.append(f"|{a}|{fmt(v)}|{r['supported_record_views']}/{r['required_record_views']}|"
            f"{r['known_easy_risk_violations']}|{r['undefined_easy_record_views']}|{r['selected']}|")
    lines += ['', 'The original2% full-record all/easy harm/reference and easy-degradation criteria are unchanged.',
        'Matched-count comparisons test utility only;empty comparisons are not estimable,not safe-zero outcomes.',
        '[Actual original-unit training curves](training_report.md) do not substitute for final-selection calibration.', '',
        '## Within-unit context controls', '',
        '|New-unit contrast|Full known pp [95% CI]|Full completion-lower pp [95% CI]|', '|---|---:|---:|---:|']
    for name in ('set_joint_minus_row_joint', 'set_matched_minus_row_matched'):
        v = s['comparisons'][name]; lines.append(f"|{name}|{fmt(v['full_known_pp'])}|{fmt(v['full_lower_pp'])}|")
    lines += ['', f"Nonoptimal query views: {value['nonoptimal_queries']}.",
        f"Selected-set joint decision reasons: {value['reason_counts']}.", '', '## Fitting gates', '']
    lines += [f"- {k}: {'pass' if v else 'fail'}" for k, v in s['fitting_gates'].items()]
    lines += ['', f"{sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])} fitting checks;advancement {s['fitting_advancement_gate']}.", '',
        '## Scope and limits', '', 'Exposed SOURCE fitting/resubstitution only,not independent confirmation or whole-pipeline OOF.',
        'All149,510 unique target rows,18,766 current-frame queries,74 recordings and12 localities remain.',
        'Repeated views and overlapping windows are not independent observations. ADE uses available labels;',
        'full-roster FDE is not estimable for missing endpoints. Wholly-unknown bounds do not certify missing interior labels.',
        'Training uses original selected sets;final allocation can select a different set. Normalizing a loss correctly',
        'does not guarantee final-selection calibration or finite-record risk. No risk threshold or decoder constraint is relaxed.',
        'The same seven forecasts remain fixed;this experiment trains risk critics,not new trajectory dynamics.',
        'Predicted-path geometry is a proxy,not verified physical safety or all-visible-agent modeling.',
        'Task:obs8/pred12 at raw stride12,detector-silver/local-image coordinates,not historical raw-frame t+50.',
        'No metric,seconds-level,human-gold,true3D,foundation,independent-success or submission-ready claim.',
        'Deployment stays unchanged;independent roles,Stage5C execution and SMC remain closed.', '']
    return '\n'.join(lines)


def complete_report():
    value = build(); p = run.PUBLIC
    pack(ROOT, p); verify(ROOT, p)
    run.once(p/'replay_runtime.json', value['replay_runtime']); (p/'readout.md').write_text(render(value))
    run.write(p/'mechanism_diagnostics.json', {k: value[k] for k in ('nonoptimal_queries', 'reason_counts')})
    policy_plot(value)
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), training_freeze_sha256=run.sha(p/'training_freeze.json'),
        report_code_sha256=run.sha(Path(__file__)), report_sha256=run.sha(p/'readout.md'),
        policy_figure_sha256=run.sha(p/'figures/query_unit_policy.svg'),
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, gates=value['summary']['fitting_gates'], deployment_changed=False)))


def policy_plot(value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8)); s = value['summary']
    names = ('old_units_row_joint', 'row_joint', 'old_units_set_joint', 'set_joint')
    labels = ('Old units / local', 'New units / local', 'Old units / set', 'New units / set')
    for i, name in enumerate(names):
        old = name.startswith('old_units_')
        v = (s['old_units_accuracy'] if old else s['accuracy'])[name]
        risk = (s['old_units_safety'] if old else s['safety'])[name]
        color = '#666666' if old else '#007C91'; low, high = v['CI95']
        axes[0].errorbar(v['point'], i, xerr=[[max(0, v['point']-low)], [max(0, high-v['point'])]],
            fmt='o', capsize=3, color=color)
        axes[2].barh(i-.14, risk['supported_record_views'], height=.27, color='#007C91')
        axes[2].barh(i+.14, risk['known_easy_risk_violations'], height=.27, color='#B44D48')
    for i, (arm, field, label) in enumerate((('row', 'full_known_pp', 'Local / known'),
            ('row', 'full_lower_pp', 'Local / lower'), ('set', 'full_known_pp', 'Set / known'),
            ('set', 'full_lower_pp', 'Set / lower'))):
        v = s['query_unit_contrasts'][arm][field]; low, high = v['CI95']
        axes[1].errorbar(v['point'], i, xerr=[[max(0, v['point']-low)], [max(0, high-v['point'])]],
            fmt='o', capsize=3, color='#007C91' if field == 'full_known_pp' else '#B44D48')
    for ax in (axes[0], axes[2]): ax.set_yticks(range(4), labels); ax.invert_yaxis()
    axes[1].set_yticks(range(4), ('Local / known', 'Local / lower', 'Set / known', 'Set / lower')); axes[1].invert_yaxis()
    axes[0].set_xlabel('Known ADE gain over floor (%)'); axes[0].set_title('Full-policy accuracy')
    axes[1].set_xlabel('Paired new - old gain (pp)'); axes[1].set_title('Known vs unknown-completion lower')
    for ax in axes[:2]: ax.axvline(0, color='#888888', linewidth=.8); ax.grid(axis='x', alpha=.2)
    axes[2].set_xlim(0, 450); axes[2].set_xlabel('Recording-risk views (444 required)')
    axes[2].legend(handles=[Patch(color='#007C91', label='Supported'), Patch(color='#B44D48', label='Known easy violation')],
        loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=7, frameon=False)
    fig.suptitle('Exposed SOURCE fitting: higher known utility does not pass risk', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); p = run.PUBLIC/'figures/query_unit_policy.svg'
    fig.savefig(p, metadata={'Date': None}, bbox_inches='tight')
    p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'query_unit_policy.png', dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('training', 'complete'))
    args = parser.parse_args(); export_training() if args.phase == 'training' else complete_report()
