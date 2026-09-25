"""Verify and report every registered reference-target arm, without selection."""
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_floor_relative as run
from scripts.report_m3w_european_protected_motion import compact_metric

VARIANTS = ('cv_targets', 'floor_utility', 'floor_risk', 'floor_both', 'old_rebased')
FAMILIES = ('ADE_vs_floor', 'ADE_vs_CV', 'ADE_vs_matched_cv_target')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    if path.stat().st_size >= 1024**2:
        raise ValueError('Public aggregate exceeds 1MiB: '+str(path))


def spread(metrics):
    known = [r for r in metrics if r['equal_scene_gain_percent'] is not None]
    vals = [r['equal_scene_gain_percent'] for r in known]
    return dict(views=len(metrics), defined=len(known), range=[min(vals), max(vals)] if vals else None,
        positive_points=sum(v > 0 for v in vals),
        positive_CI=sum(r['scene_bootstrap_ci95'] is not None and r['scene_bootstrap_ci95'][0] > 0 for r in known),
        negative_CI=sum(r['scene_bootstrap_ci95'] is not None and r['scene_bootstrap_ci95'][1] < 0 for r in known),
        worst_locality=min((r['worst_scene_gain_percent'] for r in known
                            if r['worst_scene_gain_percent'] is not None), default=None))


def policy_summary(views):
    out = {family: {s: spread([r[family][s] for r in views]) for s in views[0][family]} for family in FAMILIES}
    out['FDE_vs_floor'] = spread([r['FDE_vs_floor'] for r in views])
    easy = [r['ADE_vs_CV']['easy']['worst_scene_gain_percent'] for r in views]
    out['safety'] = dict(worst_positive_easy_degradation_percent=-min(v for v in easy if v is not None),
        zero_CV_harm_views=sum(r['zero_CV']['harmed_rows'] > 0 for r in views),
        unknown_ADE_switches_per_view=[r['unknown_ADE_switches'] for r in views],
        switch_rate_range=[min(r['switch_rate'] for r in views), max(r['switch_rate'] for r in views)])
    return out


def verify():
    run.ensure_both_frozen()
    cfg = json.loads((ROOT/run.CONFIG).read_text())
    if run.digest(run.DIAG/'summary_metrics.json') != cfg['prior_floor_summary_sha256']:
        raise ValueError('Bound prior population and diagnostic changed')
    modes, refs, heads = {}, {}, {}
    counts = dict(saved_decision_arrays_verified=0, independent_coordinate_arrays=0,
                  independent_metric_reductions=0, matched_old_metrics=0)
    for mode in ('batch', 'fitting'):
        base = run.PRIVATE/mode
        identity = json.loads((base/'identity.json').read_text())
        for key, path in (('cross_analysis_sha256', 'analysis.json'),):
            run.cross.configure(mode)
            if run.digest(run.cross.PUBLIC/path) != identity[key]:
                raise ValueError('Changed predecessor analysis')
        for key, path in (('cross_identity_sha256', 'identity.json'),
                          ('cross_decisions_sha256', 'decisions_complete.json')):
            if run.digest(run.cross.PRIVATE/path) != identity[key]:
                raise ValueError('Changed predecessor decision provenance')
        if run.digest(run.cross.parent.PRIVATE/'identity.json') != identity['common_parent_sha256']:
            raise ValueError('Changed common producer identity')
        inner = json.loads((base/'inner_complete.json').read_text())
        trained = json.loads((base/'training_complete.json').read_text())
        done = json.loads((base/'evaluation_complete.json').read_text())
        if (inner['identity'] != identity or trained['identity'] != identity or done['identity'] != identity
                or not inner['source_exclusion_pass'] or not done['all_passed']
                or len(inner['archives']) != 36 or len(trained['heads']) != 72 or len(done['groups']) != 18):
            raise ValueError('Complete matched training and evaluation required')
        hr = list(trained['heads'])
        for ref in inner['archives']:
            if run.artifact(ROOT/ref['path']) != ref:
                raise ValueError('Inner receipt changed')
            r = json.loads((ROOT/ref['path']).read_text()); p = r['provenance']; l = p['lineage']
            run.assert_producer_exclusion(l['training_sites'], l['target_sites'], l['outer_sites'], l['fitting_parent'])
            if len(l['outer_sites']) != 8 or p['identity'] != identity or run.artifact(ROOT/r['artifact']['path']) != r['artifact']:
                raise ValueError('Incorrect floor producer or changed causal decisions')
            hr.extend((p['utility'], p['risk']))
        for ref in hr:
            if run.artifact(ROOT/ref['path']) != ref:
                raise ValueError('Head completion receipt changed')
            if ref['path'] not in heads:
                r = json.loads((ROOT/ref['path']).read_text())
                heads[ref['path']] = run.checked((ROOT/ref['path']).parent, r['identity'])
        rows = {}
        for ref in done['groups']:
            if run.artifact(ROOT/ref['path']) != ref:
                raise ValueError('Evaluation receipt changed')
            r = json.loads((ROOT/ref['path']).read_text())
            if r['identity'] != identity or not r['verified'] or set(r['views']) != set(VARIANTS):
                raise ValueError('Unverified or missing factorial arm')
            for v in r['views'].values():
                m = v['ADE_vs_floor']['all']
                if len(m['expected_scenes']) != 8 or set(m['by_scene']) != set(m['expected_scenes']):
                    raise ValueError('Complete eight-locality roster required')
            for k in counts:
                counts[k] += r[k]
            rows[r['group']] = r
        if len(rows) != 18:
            raise ValueError('Duplicate group')
        modes[mode], refs[mode] = rows, done['groups']
    if len(heads) != 234 or sum(h['fit']['step'] for h in heads.values()) != 468000:
        raise ValueError('Registered full training budget not completed')
    if any(h['fit']['unknown_rows_sampled'] for h in heads.values()):
        raise ValueError('Unknown target entered supervised sampling')
    if counts != dict(saved_decision_arrays_verified=144, independent_coordinate_arrays=144,
                       independent_metric_reductions=2160, matched_old_metrics=180):
        raise ValueError('Independent checks incomplete')
    events = [json.loads(line) for line in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    first_eval = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    frozen = {m: min(e['utc'] for e in events if e['state'] == 'phase_complete'
                     and e.get('phase') == 'decide' and e.get('mode') == m) for m in modes}
    if any(t >= first_eval for t in frozen.values()):
        raise ValueError('Both decisions must precede first evaluation')
    return modes, refs, heads, counts, dict(decisions_completed=frozen, first_evaluation=first_eval)


def publish_mode(mode, rows):
    pub = run.PUBLIC/mode; pub.mkdir(parents=True, exist_ok=True)
    compact = {}
    for group, r in rows.items():
        compact[group] = {}
        for arm, v in r['views'].items():
            compact[group][arm] = {family: {s: compact_metric(m) for s, m in v[family].items()} for family in FAMILIES}
            compact[group][arm].update(FDE_vs_floor=compact_metric(v['FDE_vs_floor']),
                **{k: v[k] for k in ('switch_rate', 'switched_rows', 'unknown_ADE_switches', 'zero_CV')})
    dump(pub/'view_metrics.json', compact)
    for fold in range(3):
        with (pub/f'fold{fold}_locality_metrics.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99', 'reference_p95')
            w.writerow(['group', 'arm', 'metric', 'subset', 'locality', *fields])
            for group, r in rows.items():
                if not group.startswith(f'fold{fold}_'):
                    continue
                for arm, v in r['views'].items():
                    mm = [(family, s, m) for family in FAMILIES for s, m in v[family].items()]
                    mm.append(('FDE_vs_floor', 'all', v['FDE_vs_floor']))
                    for family, s, m in mm:
                        for site, val in m['by_scene'].items():
                            w.writerow([group, arm, family, s, site, *[val.get(k) for k in fields]])
    return {event: {arm: policy_summary([r['views'][arm] for group, r in rows.items() if group.endswith('_'+event)])
                   for arm in VARIANTS} for event in ('all', 'easy')}


def training_summary(heads):
    rows = []
    for path, r in heads.items():
        f = r['fit']; fixed = f.get('fixed_trace', [])
        rows.append(dict(head=path.removeprefix(str(run.PRIVATE.relative_to(ROOT))+'/').removesuffix('/complete.json'),
            kind=r['kind'], steps=f['step'], resumed_phase_updates=f['new_updates'], seconds=f['seconds'],
            parameters=f['parameters'], draws=f['total_draws'], unique_rows=f['unique_training_rows'],
            checkpoint_replay_rows=r['replay_rows'], unknown_sampled=f['unknown_rows_sampled'],
            fixed_start_loss=fixed[0]['loss'] if fixed else None,
            fixed_end_loss=fixed[-1]['loss'] if fixed else None,
            stochastic_start_loss=f['trace'][0]['loss'], stochastic_end_loss=f['trace'][-1]['loss']))
    with (run.PUBLIC/'training_inventory.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    pilot = json.loads((run.PRIVATE/'pilot.json').read_text())['fit']
    actual_updates = sum(r['resumed_phase_updates'] for r in rows)+pilot['new_updates']
    if actual_updates != 468000 or pilot['step'] != 100:
        raise ValueError('Pilot/resume update accounting inconsistent')
    fixed = [r for r in rows if r['fixed_start_loss'] is not None]
    return dict(heads=len(rows), updates=actual_updates, pilot_updates=pilot['step'],
        parameter_range=[min(r['parameters'] for r in rows), max(r['parameters'] for r in rows)],
        summed_fit_seconds=sum(r['seconds'] for r in rows)+pilot['seconds'],
        fixed_training_loss_heads=len(fixed), fixed_training_loss_decreased=sum(r['fixed_end_loss'] < r['fixed_start_loss'] for r in fixed),
        total_draws=sum(r['draws'] for r in rows), checkpoint_replays=len(rows),
        note='Risk fixed-training losses are not validation; utility losses use changing minibatches. Fit seconds exclude I/O and evaluation.')


def zero_context(data, designs):
    """Outcome-filtered forensic summary, never supplied to an inference head."""
    cv = data['baseline_ade'][:, 1]
    zero = np.isfinite(cv) & (cv == 0)
    h, valid = data['history'][zero], data['valid'][zero]
    speed = np.linalg.norm(np.diff(h.astype(float), axis=1), axis=2)
    return dict(result_source='fresh_run_posthoc_forensics_not_a_new_policy', rows=int(zero.sum()),
        localities=len(set(data['sites'][zero])), valid_future_counts=valid.sum(1).tolist(),
        endpoint_supported=int(valid[:, -1].sum()), last_step_stationary=int((speed[:, -1] == 0).sum()),
        accepted_by_existing_past_motion_guard=int((speed.sum(1) > 0).sum()),
        by_fold={str(fold): dict(fitting_zero_rows=int(zero[d['train_ids']].sum()),
                                held_zero_rows=int(zero[d['held_ids']].sum())) for fold, d in designs.items()},
        caveat='Outcome-defined subset is diagnostic only. Partial future labels do not prove full-horizon stationarity or safety.')


def posthoc_support():
    _, ctx, _, _ = run.load('batch')
    designs = {fold: design for _, candidate, fold, seed, design
               in run.cross.parent.jobs(ctx[1], ctx[2], ctx[3]['parent_identity']) if candidate == 'neural'}
    r = zero_context(ctx[2], designs)
    dump(run.PUBLIC/'zero_reference_context.json', r)
    return r


def figures(modes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, (mode, rows) in enumerate(modes.items()):
        for row, event in enumerate(('all', 'easy')):
            ax = axes[row, col]; rr = [r for g, r in rows.items() if g.endswith('_'+event)]
            for offset, arm, color, label in ((-.14, 'cv_targets', '#b34248', 'Matched CV targets'),
                    (.14, 'floor_both', '#13766c', 'Floor-relative targets')):
                vals = [r['views'][arm]['ADE_vs_floor']['all'] for r in rr]
                x = np.array([v['equal_scene_gain_percent'] for v in vals]); ci = np.array([v['scene_bootstrap_ci95'] for v in vals])
                ax.errorbar(x, np.arange(9)+offset, xerr=[x-ci[:, 0], ci[:, 1]-x], fmt='o', color=color,
                            markersize=4, label=label)
            ax.axvline(0, color='#777777', linewidth=.8)
            ax.set_yticks(np.arange(9), [r['group'].replace('_'+event, '') for r in rr], fontsize=8)
            ax.set_title(f'{mode} normalizer / {event} event')
            ax.set_xlabel('All-ADE gain over protected floor (%)'); ax.grid(axis='x', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .92), ncol=2, frameon=False)
    fig.suptitle('Matched reference-target ablation\nOpened development; conditional locality intervals, no deployment', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .88])
    svg = run.PUBLIC/'reference_targets.svg'; fig.savefig(svg)
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'reference_targets.png', dpi=140); plt.close(fig)


def loss_figure(heads):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, mode in enumerate(('batch', 'fitting')):
        for row, event in enumerate(('all', 'easy')):
            ax = axes[row, col]
            for ref, color in (('cv', '#b34248'), ('floor', '#13766c')):
                curves = [r['fit']['fixed_trace'] for p, r in heads.items()
                          if f'/{mode}/heads/' in p and p.endswith(f'_{event}_{ref}_risk/complete.json')]
                assert len(curves) == 9
                steps = [p['step'] for p in curves[0]]
                assert all([p['step'] for p in c] == steps for c in curves)
                y = np.array([[p['loss']/c[0]['loss'] for p in c] for c in curves])
                ax.plot(steps, np.median(y, axis=0), color=color, label=ref+' target')
                ax.fill_between(steps, np.min(y, axis=0), np.max(y, axis=0), color=color, alpha=.1)
            ax.set_title(f'{mode} normalizer / {event} event')
            ax.set_xlabel('Optimizer update'); ax.set_ylabel('Loss / initial fixed-batch loss')
            ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.suptitle('Risk-head optimization: median and full range across nine fits\nFixed fitting minibatch only; not validation or evidence of generalization', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .86])
    svg = run.PUBLIC/'training_losses.svg'; fig.savefig(svg)
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'training_losses.png', dpi=140); plt.close(fig)


def main():
    modes, refs, heads, counts, barrier = verify()
    summary = {mode: publish_mode(mode, rows) for mode, rows in modes.items()}
    training = training_summary(heads)
    zero = posthoc_support()
    population = json.loads((run.DIAG/'summary_metrics.json').read_text())
    dump(run.PUBLIC/'summary_metrics.json', dict(result_source='fresh_run_training_and_evaluation', modes=summary,
        training=training, checks=counts, evaluation_barrier=barrier, new_forecaster_training=False,
        zero_reference_context=zero,
        inner_floor_producer_sites=2, held_floor_producer_sites=4, producer_size_shift=True,
        unique_opened_development_localities=population['unique_localities'],
        indexed_overlapping_windows=population['primary_population_rows'],
        population_status='cached_verified_same_bound_prior_population',
        independent_confirmation=False, reserved_roles_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    lines = ['# Floor-Relative Target Experiment', '',
        'Every preregistered arm is retained. All values are equally weighted locality gains (%),',
        'not pooled-window gains. Each group uses eight held-out development localities.', '',
        '| Mode | Group | Arm | All vs floor | Hard vs floor | Easy vs CV | All vs matched CV targets | Switch % | Zero-CV harmed rows |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for mode, rows in modes.items():
        for group, r in rows.items():
            for arm, v in r['views'].items():
                vals = [v['ADE_vs_floor']['all'], v['ADE_vs_floor']['hard'], v['ADE_vs_CV']['easy'], v['ADE_vs_matched_cv_target']['all']]
                lines.append('| '+mode+' | '+group+' | '+arm+' | '+' | '.join(f"{m['equal_scene_gain_percent']:.6f}" for m in vals)
                    +f" | {100*v['switch_rate']:.4f} | {v['zero_CV']['harmed_rows']} |")
    lines += ['', 'Full conditional 3,000-locality-bootstrap intervals, annotation support, tail errors,',
        'and worst localities are in the aggregate JSON and per-fold CSV files. Three seeds and',
        'overlapping folds are dependent development views, not independent replications.',
        'Raw observation/prediction protocol: 8/12 at annotation stride12; image pixels.',
        'No raw-t50 equivalence, metric/seconds, human gold, physical safety, true3D or foundation claim.',
        'No deployment, reserved-role opening, Stage5C execution or SMC.', '']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)); figures(modes); loss_figure(heads)
    tests = json.loads((run.PRIVATE/'test_receipt.json').read_text()); log = run.PRIVATE/'tests.txt'
    passed = re.search(r'(\d+) passed', log.read_text())
    if tests['returncode'] != 0 or run.digest(log) != tests['log_sha256'] or passed is None:
        raise ValueError('Passing scoped test receipt required')
    paths = sorted(p for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'completion_checks.json')
    for p in paths:
        if p.stat().st_size >= 1024**2: raise ValueError('Oversized public file: '+str(p))
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, group_receipts=refs,
        heads_verified=len(heads), checks=counts, summary_sha256=run.digest(run.PUBLIC/'summary_metrics.json'),
        aggregate_sha256={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in paths},
        reporter_sha256=run.digest(Path(__file__)), tests_passed=int(passed.group(1)),
        scoped_test_files=tests['files'], test_log_sha256=tests['log_sha256'], full_legacy_suite_run=False,
        training_complete=True, independent_confirmation=False, deployment_changed=False))
    print(json.dumps(dict(groups=36, heads=len(heads), checks=counts, training=training)))


if __name__ == '__main__':
    main()
