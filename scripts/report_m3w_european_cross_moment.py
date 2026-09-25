"""Publish both cross-moment controls without selecting a readout winner."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cross_moment as run
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_hurdle_coverage import counts


def name(value):
    return value.replace('product', 'control').replace('hurdle', 'treatment').replace('ranked', 'treatment')


def rename(value):
    if isinstance(value, dict):
        return {name(k): rename(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename(v) for v in value]
    return value


def verify():
    a = json.loads((run.PUBLIC/'analysis.json').read_text())
    v = json.loads((run.PUBLIC/'verification.json').read_text())
    h = json.loads((run.PUBLIC/'head_replay.json').read_text())
    if (not v['all_passed'] or v['analysis_sha256'] != run.digest(run.PUBLIC/'analysis.json')
            or v['identity'] != a['identity'] or h['identity'] != a['identity'] or not h['all_passed']
            or len(h['checks']) != 36 or v['scalar_sorting_checks'] != 216
            or v['separate_coordinate_reductions'] != 864 or v['separate_decompositions'] != 108
            or a['parent_controls_reproduced'] != 36 or len(a['views']) != 216):
        raise ValueError('Complete fitted-head, causal-sort and metric verification required')
    support = json.loads((run.PUBLIC/'fitting_support.json').read_text())
    if support['identity'] != a['identity'] or not support['fitting_only'] or len(support['rows']) != 36:
        raise ValueError('Fitting-only support audit required')
    run.ensure_all_decisions()
    if a['control_analysis_sha256'] != run.digest(run.control_analysis_path(a['identity'])):
        raise ValueError('Preceding control analysis changed')
    run.assert_identity(a['identity']); run.read_decisions(a['identity'])
    return a


def fmt(x):
    return 'undefined' if x is None else f'{x:.6f}'


def reports(a):
    pub = run.PUBLIC
    rows = ['# Cross-Moment '+run.MODE+': Every Registered View', '',
        'Batch mode changes pair weighting; fitting mode changes only its normalizer.',
        'Control: '+str(a['internal_helper_aliases'])+'.',
        'Treatment is cross-moment ranking. All count controls are offline diagnostics.', '',
        '| View | ADE gain vs CV (%) | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Predicted violations |',
        '|---|---:|---:|---:|---:|---|---:|---:|']
    views = {}
    for key, r in a['views'].items():
        easy = r['ADE_vs_CV']['easy']['worst_scene_gain_percent']
        vals = [r['ADE_vs_CV']['all']['equal_scene_gain_percent'], r['FDE_vs_CV']['equal_scene_gain_percent'],
                r['ADE_vs_CV']['hard']['equal_scene_gain_percent'], -easy if easy is not None else None]
        rows.append(f"| {name(key)} | {' | '.join(fmt(v) for v in vals)} | {r['zero_CV']['harmed_rows']}/{r['zero_CV']['rows']} | {100*r['switch_rate']:.6f} | {r['predicted_risk_violations']} |")
        views[name(key)] = {k: r[k] for k in ('switch_rate', 'selected_rows', 'selected_unknown_ADE',
            'selected_unknown_FDE', 'predicted_risk_violations', 'zero_CV', 'observed_preservation', 'decision_sha256')}
        views[name(key)].update(ADE_vs_CV={s: compact_metric(m) for s, m in r['ADE_vs_CV'].items()},
                                FDE_vs_CV=compact_metric(r['FDE_vs_CV']))
    rows += ['', '## Ranking and Coverage Components', '',
        'Both anchors retained. Positive ranks favor the new ordering. Units: pp of CV-normalized ADE.',
        'Exact accounting, not unique causal mediation. Common-pool and full-anchor effects both retained.', '',
        '| Group | Subset | Component | Mean (pp) | Conditional 95% CI (pp) |', '|---|---|---|---:|---|']
    components = ('full_total', 'support_difference', 'total', 'ranking_at_product_count',
                  'ranking_at_hurdle_count', 'coverage_with_hurdle_ranking', 'coverage_with_product_ranking')
    with (pub/'contrasts.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['group', 'subset', 'component', 'mean_pp', 'low_pp', 'high_pp'])
        for key, subsets in a['decomposition'].items():
            for subset, value in subsets.items():
                for component in components:
                    v = value[component]; mean, ci = (None, [None, None]) if v is None else (v['mean_gain_difference_pp'], v['ci95_pp'])
                    w.writerow([key, subset, name(component), mean, *ci])
                    rows.append(f"| {key} | {subset} | {name(component)} | {fmt(mean)} | [{fmt(ci[0])}, {fmt(ci[1])}] |")
    groups, safety = {}, {}
    for candidate in ('neural', 'damping097'):
        groups[candidate] = {}; safety[candidate] = {}
        for event in ('all', 'easy'):
            d = [r for k, r in a['decomposition'].items() if k.startswith(candidate+'_') and k.endswith('_'+event)]
            assert len(d) == 9
            groups[candidate][event] = {s: {name(c): counts([r[s][c] for r in d]) for c in components}
                                        for s in ('all', 'easy', 'hard')}
            safety[candidate][event] = {}
            for arm in run.ARMS:
                vs = [r for k, r in a['views'].items() if k.startswith(candidate+'_') and k.endswith('_'+event+'_'+arm)]
                assert len(vs) == 9
                safety[candidate][event][name(arm)] = dict(views=9,
                    observed_preservation=sum(v['observed_preservation'] for v in vs),
                    easy_failures=sum(v['ADE_vs_CV']['easy']['worst_scene_gain_percent'] < -2 for v in vs),
                    worst_easy_degradation_percent=max(-v['ADE_vs_CV']['easy']['worst_scene_gain_percent'] for v in vs),
                    zero_reference_harm_views=sum(v['zero_CV']['harmed_rows'] > 0 for v in vs),
                    predicted_violation_views=sum(v['predicted_risk_violations'] > 0 for v in vs),
                    switch_percent_range=[100*min(v['switch_rate'] for v in vs), 100*max(v['switch_rate'] for v in vs)])
    versus = {name(k): {s: compact_metric(m) for s, m in r.items()} for k, r in a['neural_vs_damping'].items()}
    vstats = {}
    for arm in ('control_original', 'ranked_original'):
        vstats[name(arm)] = {}
        for subset in ('all', 'easy', 'hard'):
            ms = [r[subset] for k, r in a['neural_vs_damping'].items() if k.endswith('_'+arm)]
            assert len(ms) == 18
            vstats[name(arm)][subset] = dict(comparisons=len(ms),
                positive_points=sum(v['equal_scene_gain_percent'] > 0 for v in ms),
                positive_CI=sum(v['scene_bootstrap_ci95'][0] > 0 for v in ms),
                negative_CI=sum(v['scene_bootstrap_ci95'][1] < 0 for v in ms),
                gain_percent_range=[min(v['equal_scene_gain_percent'] for v in ms), max(v['equal_scene_gain_percent'] for v in ms)])
    for family in ('ade', 'fde'):
        with (pub/f'locality_{family}_metrics.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99')
            w.writerow(['view', 'subset', 'locality', *fields])
            for key, r in a['views'].items():
                metrics = r['ADE_vs_CV'] if family == 'ade' else {'FDE': r['FDE_vs_CV']}
                for subset, m in metrics.items():
                    for site, v in m['by_scene'].items():
                        w.writerow([name(key), subset, site, *[v.get(f) for f in fields]])
    training = {}
    for key in a['decomposition']:
        r = run.checked(key, a['identity'])
        fixed = r['fit']['fixed_trace']
        if (fixed[0]['step'] != 0 or fixed[-1]['step'] != 2000
                or len({v['batch_sha256'] for v in fixed}) != 1):
            raise ValueError('Fixed fitting diagnostic changed')
        training[key] = dict(fit=r['fit'], checkpoint=r['artifacts']['checkpoint'])
    tot = dict(heads=len(training), updates=sum(r['fit']['step'] for r in training.values()),
        fit_seconds=sum(r['fit']['seconds'] for r in training.values()),
        rank_pair_draws=sum(r['fit']['rank_pairs_seen'] for r in training.values()),
        unknown_training_draws=sum(r['fit']['unknown_rows_sampled'] for r in training.values()),
        parameters_per_head=sorted(set(r['fit']['parameters'] for r in training.values())))
    assert tot['heads'] == 36 and tot['updates'] == 72000 and tot['unknown_training_draws'] == 0
    for file, value in [('view_metrics.json', views), ('group_metrics.json', groups), ('safety_summary.json', safety),
                        ('neural_vs_damping.json', versus), ('comparison_summary.json', vstats),
                        ('training_metrics.json', dict(totals=tot, heads=training)),
                        ('decomposition_metrics.json', rename(a['decomposition']))]:
        (pub/file).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    training_diagnostics(a, training)
    rows += ['', 'Three seeds, 3,000 paired locality resamples, dependent unadjusted development views.',
        'Detector-track image pixels, obs8/pred12 rawstride12. Not independent confirmation, seconds, metric,',
        'human gold, physical safety, true 3D or foundation. No deployment, Stage5C or SMC.', '']
    (pub/'results.md').write_text('\n'.join(rows))



def training_diagnostics(a, training):
    support = json.loads((run.PUBLIC/'fitting_support.json').read_text())['rows']
    summary = {}
    lines = ['# Fitting Support and Fixed-Batch Diagnostics', '',
        'Paired examples are repeated training draws, not independent observations.',
        'Weight effective pair count is (sum w)^2/sum(w^2), not an independent sample size.',
        'Fixed batches are cloned from the original sampler and never used for selection.', '',
        '| Candidate / event | Share / cross pairs, first 40 fitting batches | Control / treatment full-fit pairs | Fixed total loss decreased | Fixed ranking loss decreased |',
        '|---|---:|---:|---:|---:|']
    for candidate in ('neural', 'damping097'):
        summary[candidate] = {}
        for event in ('all', 'easy'):
            keys = [k for k in training if k.startswith(candidate+'_') and k.endswith('_'+event)]
            old = [run.control_checked(k, a['identity'])['fit'] for k in keys]
            fit = [training[k]['fit'] for k in keys]
            audit = [support[k] for k in keys]
            out = dict(heads=len(keys), updates=sum(r['step'] for r in fit),
                old_full_pair_draws=sum(r['rank_pairs_seen'] for r in old),
                new_full_pair_draws=sum(r['rank_pairs_seen'] for r in fit),
                old_audit_pairs=sum(r['old_pairs'] for r in audit),
                new_audit_pairs=sum(r['new_pairs'] for r in audit),
                old_zero_pair_audit_batches=sum(r['old_zero_batches'] for r in audit),
                new_zero_pair_audit_batches=sum(r['new_zero_batches'] for r in audit),
                fixed_total_loss_decreased=sum(r['fixed_trace'][-1]['loss'] < r['fixed_trace'][0]['loss'] for r in fit),
                fixed_rank_loss_decreased=sum(r['fixed_trace'][-1]['ranking_loss'] < r['fixed_trace'][0]['ranking_loss'] for r in fit))
            out['cross_weight_summary'] = dict(
                fixed_normalizer_range=[min(r['mean_cross_weight_sum'] for r in audit), max(r['mean_cross_weight_sum'] for r in audit)],
                largest_pair_fraction=max(r['max_pair_weight_fraction'] for r in audit),
                mean_effective_pair_range=[min(r['mean_effective_pairs'] for r in audit), max(r['mean_effective_pairs'] for r in audit)],
                audit_p99_sum_range=[min(r['cross_weight_sum_quantiles'][2] for r in audit), max(r['cross_weight_sum_quantiles'][2] for r in audit)])
            logged = [v for r in fit for v in r['trace']]
            out['sampled_optimizer_diagnostics'] = dict(
                logged_steps=len(logged),
                logged_preclip_gradient_above_5=sum(v['gradient_norm'] > 5 for v in logged),
                largest_logged_preclip_gradient=max(v['gradient_norm'] for v in logged),
                logged_ranking_loss_range=[min(v['ranking_loss'] for v in logged), max(v['ranking_loss'] for v in logged)],
                scope='logged steps only; not the complete gradient or optimizer-step distribution')
            out['fixed_loss_endpoints'] = {k: {m: [training[k]['fit']['fixed_trace'][j][m] for j in (0, -1)]
                for m in ('loss', 'ranking_loss', 'moment_mse', 'bce', 'conditional_mse')} for k in keys}
            summary[candidate][event] = out
            lines.append(f"| {candidate} / {event} | {out['old_audit_pairs']} / {out['new_audit_pairs']} | "
                         f"{out['old_full_pair_draws']} / {out['new_full_pair_draws']} | "
                         f"{out['fixed_total_loss_decreased']}/9 | {out['fixed_rank_loss_decreased']}/9 |")
    lines += ['', 'Audit old/new counts compare share/cross targets in both modes, not the preceding normalizer.',
        'A lower fitting loss is not proof of better excluded-locality ordering.',
        'Only the paired full and matched-count readout can establish that effect.', '']
    (run.PUBLIC/'pair_support_summary.json').write_text(json.dumps(summary, separators=(',', ':'), allow_nan=False)+'\n')
    (run.PUBLIC/'fitting_diagnostics.md').write_text('\n'.join(lines))


def complete():
    identities, analyses, pids, aggregates, first_decisions, readouts, summary = {}, {}, [], {}, {}, [], {}
    for mode in ('batch', 'fitting'):
        run.configure(mode)
        a = verify()
        analyses[mode] = run.digest(run.PUBLIC/'analysis.json')
        identities[mode] = a['identity']
        aggregates[mode] = {f: run.digest(run.PUBLIC/f) for f in (
            'view_metrics.json', 'group_metrics.json', 'safety_summary.json',
            'neural_vs_damping.json', 'comparison_summary.json', 'training_metrics.json',
            'decomposition_metrics.json', 'pair_support_summary.json',
            'fitting_support.json', 'verification.json', 'head_replay.json')}
        summary[mode] = {k: json.loads((run.PUBLIC/f).read_text()) for k, f in (
            ('groups', 'group_metrics.json'), ('safety', 'safety_summary.json'),
            ('neural_vs_damping', 'comparison_summary.json'), ('fitting', 'pair_support_summary.json'))}
        summary[mode]['training_totals'] = json.loads((run.PUBLIC/'training_metrics.json').read_text())['totals']
        events = [json.loads(v) for v in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
        first_decisions[mode] = {}
        for v in events:
            if v['state'] == 'decisions_frozen':
                first_decisions[mode].setdefault(v['group'], v['utc'])
            if v['state'] == 'group_evaluated':
                readouts.append(v['utc'])
        last = {v['pid']: v for v in events}
        for pid, v in last.items():
            proc = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
            if v['state'] != 'phase_complete' or 'run_m3w_european_cross_moment.py' in proc.stdout:
                raise ValueError('Required phase not terminal')
        pids.extend(last)
    if any(len(v) != 36 for v in first_decisions.values()) or not readouts:
        raise ValueError('Complete decision/readout event history required')
    last_frozen = max(t for v in first_decisions.values() for t in v.values())
    first_readout = min(readouts)
    if last_frozen > first_readout:
        raise ValueError('Observed readout preceded full decision freeze')
    _, _, data, _, _ = run.load()
    zero = np.isfinite(data['baseline_ade'][:, 1]) & (data['baseline_ade'][:, 1] == 0)
    zero_support = dict(rows=int(zero.sum()),
        by_locality={str(s): int((zero & (data['sites'] == s)).sum()) for s in sorted(set(data['sites'][zero]))},
        valid_future_steps=data['valid'][zero].sum(1).tolist(),
        endpoint_supported=data['valid'][zero, -1].tolist())
    run.json_write(run.PUBLIC_ROOT/'summary_metrics.json', dict(
        result_source='fresh_run_two_cross_moment_controls', modes=summary,
        zero_reference_label_support=zero_support, analyses_sha256=analyses,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    parent = json.loads((run.supported.PUBLIC/'completion_checks.json').read_text())
    tests = parent['scoped_test_files'] + ['tests/test_m3w_cross_moment_rank.py',
                                         'tests/test_m3w_cross_moment_protocol.py']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT, capture_output=True, text=True)
    (run.PUBLIC_ROOT/'scoped_tests.txt').write_text(result.stdout+result.stderr)
    if result.returncode or not re.search(r'\b264 passed\b', result.stdout):
        raise ValueError('Scoped tests failed')
    run.json_write(run.PUBLIC_ROOT/'completion_checks.json', dict(
        result_source='fresh_run_cross_moment_controls_training_and_verification',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        analyses_sha256=analyses, identities=identities, aggregate_sha256=aggregates,
        summary_sha256=run.digest(run.PUBLIC_ROOT/'summary_metrics.json'),
        pre_readout_barrier=dict(last_initial_decision_utc=last_frozen,
                                first_evaluated_group_utc=first_readout, passed=True),
        reporter_sha256=run.digest(Path(__file__)),
        tests_sha256=run.digest(run.PUBLIC_ROOT/'scoped_tests.txt'),
        scoped_test_files=tests, tests_passed=264, full_legacy_suite_run=False,
        heads=72, updates=144000, views=432, old_controls=72,
        completed_pids=sorted(pids), all_required_phases_finished=True,
        reserved_roles_opened=False, deployment_changed=False, stage5c_executed=False,
        smc_enabled=False, submission_ready=False))
    print(json.dumps(dict(heads=72, updates=144000, views=432, tests=264)))


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, layout='constrained')
    rows = [(f, s, e) for f in range(3) for s in (17, 29, 43) for e in ('all', 'easy')]
    colors = ('#146b78', '#ad4936')
    for ax, component, title in zip(axes, ('ranking_at_product_count', 'ranking_at_hurdle_count'),
            ('At preceding-control common counts', 'At current-treatment common counts')):
        for i, (fold, seed, event) in enumerate(rows):
            for candidate, color, offset in zip(('neural', 'damping097'), colors, (-.14, .14)):
                v = a['decomposition'][f'{candidate}_fold{fold}_seed{seed}_{event}']['all'][component]
                if v is not None:
                    ax.plot(v['ci95_pp'], [i+offset]*2, color=color, linewidth=.9)
                    ax.plot(v['mean_gain_difference_pp'], i+offset, 'o', color=color, markersize=4)
        ax.axvline(0, color='#777777', linewidth=.8); ax.grid(axis='x', color='#dddddd', linewidth=.5)
        ax.spines[['top', 'right']].set_visible(False); ax.set_title(title, fontsize=11)
        ax.set_xlabel('Treatment advantage (pp of CV-normalized ADE)')
    axes[0].set_yticks(range(len(rows)), [f'F{f} / S{s} / {e}' for f, s, e in rows]); axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], color=c, marker='o', linestyle='none', label=n)
        for c, n in zip(colors, ('Neural candidate', 'Damping candidate'))], loc='outside lower center', ncols=2)
    fig.suptitle(f'Does cross-moment {run.MODE} improve ordering at fixed coverage?\n'
                 'All 72 contrasts; conditional locality intervals. Offline controls, not deployable policies.', fontsize=12)
    for ext in ('svg', 'png'):
        fig.savefig(run.PUBLIC/('ranking_comparison.'+ext), dpi=150)
    plt.close(fig)
    training = json.loads((run.PUBLIC/'training_metrics.json').read_text())['heads']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for key, row in training.items():
        trace = row['fit']['fixed_trace']; color = colors[0 if key.startswith('neural') else 1]
        for ax, metric in zip(axes, ('loss', 'ranking_loss')):
            ax.plot([v['step'] for v in trace], [v[metric] for v in trace], color=color,
                    linestyle='-' if key.endswith('_all') else '--', alpha=.6, linewidth=.8)
            ax.set_xlabel('Optimizer step'); ax.set_ylabel(metric); ax.grid(alpha=.2)
    fig.suptitle('All 36 fixed fitting-batch traces; not validation, no model selection')
    fig.legend(handles=[Line2D([], [], color=color, linestyle=style, label=label+' / '+event)
        for color, label in zip(colors, ('Neural', 'Damping'))
        for style, event in (('-', 'all'), ('--', 'easy'))], loc='outside lower center', ncols=4)
    for ext in ('svg', 'png'):
        fig.savefig(run.PUBLIC/('training_loss.'+ext), dpi=150)
    plt.close(fig)
    for file in run.PUBLIC.glob('*.svg'):
        file.write_text('\n'.join(line.rstrip() for line in file.read_text().splitlines())+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normalizer', choices=('batch', 'fitting'))
    p.add_argument('--complete', action='store_true')
    p.add_argument('--figures', action='store_true')
    args = p.parse_args()
    if args.normalizer:
        run.configure(args.normalizer)
        a = verify()
        reports(a)
        if args.figures:
            figures(a)
    if args.complete:
        complete()
    if not args.normalizer and not args.complete:
        p.error('Choose normalizer reports or final complete check')


if __name__ == '__main__':
    main()
