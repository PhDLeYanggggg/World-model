"""Report every producer-conditioned comparison without selecting a policy."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_producer_conditioned as run
from scripts.report_m3w_european_floor_relative import spread, dump
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_causal_abstention import fmt

POLICIES = ('global', 'producer', 'placebo', 'wrong_tag', 'legacy')
SUBSETS = ('all', 'easy', 'hard', 'complete')
COMPARISONS = ('global', 'placebo', 'wrong_tag', 'legacy')
FAMILIES = ('ADE_vs_floor4', 'ADE_vs_floor2', 'ADE_vs_old_stop4')
EXPECTED = dict(saved_decisions_verified=180, coordinate_arrays_verified=216,
                metric_reductions_verified=3402, old_anchor_metrics_exact=90)


def verify():
    run.ensure_frozen()
    cfg, _, _, _, _, identity = run.load()
    done = json.loads((run.PRIVATE/'evaluation_complete.json').read_text())
    assert done['identity'] == identity and done['all_passed'] and len(done['groups']) == 18
    rows = {}; counts = dict.fromkeys(EXPECTED, 0)
    expected = {f'half{h}__{p}' for h in (0, 1) for p in POLICIES}
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text())
        assert r['verified'] and r['identity'] == identity and set(r['views']) == expected
        for k in counts: counts[k] += r[k]
        for v in r['views'].values():
            m = v['ADE_vs_floor4']['all']
            assert len(m['expected_scenes']) == 8 and set(m['by_scene']) == set(m['expected_scenes'])
        assert set(r['contrasts']) == {'half0', 'half1'}
        assert all(set(x) == set(COMPARISONS) for x in r['contrasts'].values())
        rows[r['group']] = r
    assert len(rows) == 18 and counts == EXPECTED
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    barrier = {p: min(e['utc'] for e in events if e['state'] == 'phase_complete' and e.get('phase') == p)
               for p in ('banks', 'pilot', 'train', 'decide')}
    barrier['first_new_readout'] = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    assert barrier['banks'] < barrier['pilot'] < barrier['train'] < barrier['decide'] < barrier['first_new_readout']
    trained = run.checked_training(identity); heads = {}
    for ref in trained['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); fit = r['fit']
        assert fit['complete'] and fit['step'] == 2000 and fit['total_draws'] == 512000
        assert fit['unknown_rows_sampled'] == 0 and r['replay_exact']
        assert r['prefix_replay_rows'] == [4096, 4096]
        assert all(np.isfinite(t['loss']) and np.isfinite(t['gradient_norm']) for t in fit['trace'])
        name = Path(ref['path']).parent.name
        heads[name] = dict(arm=r['identity']['arm'], task=r['identity']['task'], event=r['identity']['event'],
            seed=r['identity']['seed'], group=r['identity']['group'], fit=fit,
            checkpoint=r['artifacts']['checkpoint'], receipt_sha256=ref['sha256'],
            label_sha256=r['identity']['label_sha256'], env_sha256=r['identity']['env_sha256'],
            fitting_ids_sha256=r['identity']['fitting_ids_sha256'])
    assert len(heads) == cfg['new_heads'] == 108
    return rows, heads, done['groups'], counts, barrier, events


def safety(views):
    gains = [v['easy_vs_CV']['worst_scene_gain_percent'] for v in views]
    gains = [g for g in gains if g is not None]
    return dict(worst_positive_easy_degradation_percent=max(0., -min(gains)) if gains else None,
        zero_CV_harm_views=sum(v['zero_CV']['harmed_rows'] > 0 for v in views),
        zero_CV_supported_views=sum(v['zero_CV']['rows'] > 0 for v in views),
        switch_rate_range=[min(v['switch_rate'] for v in views), max(v['switch_rate'] for v in views)],
        unknown_ADE_switches_range=[min(v['unknown_ADE_switches'] for v in views), max(v['unknown_ADE_switches'] for v in views)])


def summarize(views):
    return dict(**{f: {s: spread([v[f][s] for v in views]) for s in SUBSETS} for f in FAMILIES},
        FDE_vs_floor4=spread([v['FDE_vs_floor4'] for v in views]), easy_vs_CV=spread([v['easy_vs_CV'] for v in views]),
        safety=safety(views))


def paired_summary(contrasts):
    return dict(ADE={s: spread([v['ADE'][s] for v in contrasts]) for s in SUBSETS},
                FDE=spread([v['FDE'] for v in contrasts]))


def reliability_summary(views):
    records = [r for v in views for site in v['reliability'].values() for r in [site['selected']]]
    supported = [r for r in records if r['realized_harm_ratio'] is not None]
    paired = [r for r in supported if r['predicted_harm_ratio'] is not None]
    def rng(key):
        x = [r[key] for r in records if r[key] is not None]
        return [min(x), max(x)] if x else None
    return dict(dependent_locality_views=len(records), selected_supported_views=len(supported),
        empty_selection_views=sum(r['rows'] == 0 for r in records),
        realized_harm_ratio_range=rng('realized_harm_ratio'), predicted_harm_ratio_range=rng('predicted_harm_ratio'),
        realized_above_2pct=sum(r['realized_harm_ratio'] > .02 for r in supported),
        underpredicted_views=sum(r['realized_harm_ratio'] > r['predicted_harm_ratio'] for r in paired),
        semantics='Positive-harm moment ratio, not net easy degradation; dependent locality/view counts, no safety guarantee.')


def publish(rows, heads):
    for group, r in rows.items():
        compact = {}
        for name, v in r['views'].items():
            compact[name] = dict(**{f: {s: compact_metric(m) for s, m in v[f].items()} for f in FAMILIES},
                FDE_vs_floor4=compact_metric(v['FDE_vs_floor4']), easy_vs_CV=compact_metric(v['easy_vs_CV']),
                **{k: v[k] for k in ('switch_rate', 'switched_rows', 'unknown_ADE_switches', 'zero_CV', 'reliability')})
        contrasts = {h: {k: dict(ADE={s: compact_metric(m) for s, m in c['ADE'].items()}, FDE=compact_metric(c['FDE']))
                           for k, c in cs.items()} for h, cs in r['contrasts'].items()}
        dump(run.PUBLIC/'groups'/(group+'.json'), dict(views=compact, contrasts=contrasts,
            raw={h: {k: compact_metric(m, scenes=True) for k, m in v.items()} for h, v in r['raw'].items()},
            anchor={f: {s: compact_metric(m, scenes=True) for s, m in v.items()} if f == 'ADE'
                    else compact_metric(v, scenes=True) for f, v in r['anchor'].items()}))
        path = run.PUBLIC/'groups'/(group+'_localities.csv')
        with path.open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99', 'reference_p95')
            w.writerow(['policy', 'comparison', 'subset', 'locality', *fields])
            for name, v in r['views'].items():
                ms = [(fam, s, m) for fam in FAMILIES for s, m in v[fam].items()]
                ms += [('FDE_vs_floor4', 'endpoint', v['FDE_vs_floor4']), ('ADE_vs_CV', 'easy', v['easy_vs_CV'])]
                for fam, s, m in ms:
                    for site, rr in m['by_scene'].items(): w.writerow([name, fam, s, site, *[rr.get(k) for k in fields]])
            for h, cs in r['contrasts'].items():
                for other, c in cs.items():
                    for s, m in {**c['ADE'], 'endpoint_FDE': c['FDE']}.items():
                        for site, rr in m['by_scene'].items(): w.writerow([h+'__producer', other, s, site, *[rr.get(k) for k in fields]])
    pooled = {p: summarize([r['views'][f'half{h}__{p}'] for r in rows.values() for h in (0, 1)]) for p in POLICIES}
    paired = {c: paired_summary([r['contrasts'][f'half{h}'][c] for r in rows.values() for h in (0, 1)]) for c in COMPARISONS}
    strata = {event: {f'half{h}': dict(
        policies={p: summarize([r['views'][f'half{h}__{p}'] for name, r in rows.items() if name.endswith('_'+event)]) for p in POLICIES},
        contrasts={c: paired_summary([r['contrasts'][f'half{h}'][c] for name, r in rows.items() if name.endswith('_'+event)]) for c in COMPARISONS})
        for h in (0, 1)} for event in ('all', 'easy')}
    raw = {k: spread([r['raw'][f'half{h}'][k] for r in rows.values() for h in (0, 1)]) for k in ('neural_vs_full', 'floor_vs_full')}
    out = dict(result_source='fresh_run_108_neural_controller_heads_and_180_readouts',
        trajectory_forecasters='cached_verified', new_trajectory_training=False, new_controller_training=True,
        new_heads=108, updates=216000, views=180, bootstrap_resamples=3000, bootstrap_unit='source_locality',
        bootstrap_seed=39271, new_threshold_selection=False, independent_confirmation=False, deployment_changed=False,
        policies=pooled, producer_vs_controls=paired, event_and_branch=strata, raw_producer_change=raw,
        reliability={p: reliability_summary([r['views'][f'half{h}__{p}'] for r in rows.values() for h in (0, 1)]) for p in POLICIES})
    dump(run.PUBLIC/'summary_metrics.json', out)
    for name, h in heads.items(): dump(run.PUBLIC/'training'/(name+'.json'), h)
    return out


def training_summary(heads):
    result = {}
    for arm in run.ARMS:
        result[arm] = {}
        for task in ('utility', 'risk'):
            fits = [r['fit'] for r in heads.values() if r['arm'] == arm and r['task'] == task]
            first, final = [f['trace'][0]['loss'] for f in fits], [f['trace'][-1]['loss'] for f in fits]
            item = dict(heads=len(fits), updates=sum(f['step'] for f in fits), parameters=sorted(set(f['parameters'] for f in fits)),
                sampled_first_loss_range=[min(first), max(first)], sampled_final_loss_range=[min(final), max(final)],
                gradient_norm_range=[min(t['gradient_norm'] for f in fits for t in f['trace']), max(t['gradient_norm'] for f in fits for t in f['trace'])],
                unknown_rows_sampled=sum(f['unknown_rows_sampled'] for f in fits),
                optimizer_seconds_sum=sum(f['seconds'] for f in fits))
            if task == 'risk':
                fixed = [(f['fixed_trace'][0]['loss'], f['fixed_trace'][-1]['loss']) for f in fits]
                item.update(fixed_training_batch_loss_reduced_heads=sum(b < a for a, b in fixed),
                            fixed_training_batch_final_loss_range=[min(b for _, b in fixed), max(b for _, b in fixed)])
            result[arm][task] = item
    return result


def documents(summary, heads, counts, barrier, rows):
    lines = ['# Producer-Conditioned Controllers: Complete Development Readout', '',
        'I compared three newly trained, equal-capacity gain/harm controllers on identical two-source forecasts: no tag, actual producer tag, and an outcome-independent placebo tag.',
        'All 108 neural heads reached their registered 2,000 updates (216,000 total); no new trajectory forecaster was trained.',
        'All 180 views are retained. Each policy spans 36 dependent fold/seed/event/producer views, not 36 independent tests.',
        'The predictor source tag is confounded with the fitting cohort. Even a positive tag effect would not uniquely identify producer transport.', '',
        '## Identical-Forecast Primary Comparisons', '',
        '| Producer vs control | All ADE gain (%) | Positive / negative CI | Hard gain (%) | Complete-label gain (%) | FDE gain (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for c, v in summary['producer_vs_controls'].items():
        m = v['ADE']['all']; lines.append(f"| {c} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {fmt(v['ADE']['hard']['range'])} | {fmt(v['ADE']['complete']['range'])} | {fmt(v['FDE']['range'])} |")
    lines += ['', '## Common Anchors And Safety', '',
        'The four-source floor and unchanged stopping controller are common anchors. Gains over each branch\'s two-source floor are reported separately, never substituted as the main denominator.', '',
        '| Policy | All gain vs common floor4 (%) | All gain vs old stop4 (%) | Positive / negative CI vs stop4 | Worst positive-easy degradation vs CV (%) | Zero-CV harmed views | Switch rate |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p, v in summary['policies'].items():
        m = v['ADE_vs_old_stop4']['all']; s = v['safety']
        lines.append(f"| {p} | {fmt(v['ADE_vs_floor4']['all']['range'])} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {s['worst_positive_easy_degradation_percent']:.5f} | {s['zero_CV_harm_views']} | {fmt(s['switch_rate_range'])} |")
    lines += ['', '## Raw Producer Change', '', '| Comparison | All ADE gain (%) | Positive / negative CI |', '|---|---:|---:|']
    for k, m in summary['raw_producer_change'].items(): lines.append(f"| {k} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} |")
    lines += ['', '## Risk Reliability', '',
        'Predicted positive-harm ratio is not a calibrated safety certificate. Realized positive harm and net easy degradation are different quantities.', '',
        '| Policy | Supported selected locality/views | Above 2% realized harm ratio | Underpredicted | Realized ratio range | Predicted ratio range |', '|---|---:|---:|---:|---:|---:|']
    for p, r in summary['reliability'].items(): lines.append(f"| {p} | {r['selected_supported_views']} | {r['realized_above_2pct']} | {r['underpredicted_views']} | {fmt(r['realized_harm_ratio_range'])} | {fmt(r['predicted_harm_ratio_range'])} |")
    lines += ['', 'Every event, branch, fold, seed and locality is retained in groups/*.json and *_localities.csv, including tail errors.',
        'Partial trajectories only contribute observed labels; missing labels are not zero error. Complete-window ADE and endpoint FDE use their actual support.',
        'EuropeanSquares released detector tracks, image-pixel obs8/pred12, raw annotation stride12. Not historical raw-t50, metric, seconds, human gold, true3D or foundation evidence.',
        'Opened sources remain development. Independent selection/calibration/confirmation remain closed. No deployment change, Stage5C or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    train = training_summary(heads); dump(run.PUBLIC/'training_summary.json', train)
    lines = ['# Training And Reproducibility', '',
        'Protocol commit `3f5f134a1584bf0f119b9d2c83ff4b8d450e9284` was pushed before the pilot and full fit.',
        'Native arm64 CPU4/inter-op1/workers0. Each head has 382 inputs and width64. Utility has 24,642 parameters; risk has 24,707.',
        'The actual 100-step utility/risk pilot resumed inside the 2,000-step budget. Identical training draws, labels, envelopes, first380 preprocessors and fixed ranking denominators were checked across arms.',
        'All unknown-label supervised draw counts are zero. Both 4,096-row producer prefixes replay exactly from every new checkpoint (216 prefix checks).',
        'Sampled losses use changing batches. Only risk has a fixed training-batch trace; neither trace is validation or test success.', '',
        '| Arm / head | Heads | Updates | Sampled first loss | Sampled final loss | Fixed training-batch decrease |', '|---|---:|---:|---:|---:|---:|']
    for arm, tasks in train.items():
        for task, t in tasks.items(): lines.append(f"| {arm} / {task} | {t['heads']} | {t['updates']} | {fmt(t['sampled_first_loss_range'])} | {fmt(t['sampled_final_loss_range'])} | {t.get('fixed_training_batch_loss_reduced_heads', 'not measured')} |")
    lines += ['', '```json', json.dumps(dict(checks=counts, readout_barrier=barrier), indent=2), '```', '', '```sh']
    lines += [f'.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase {phase}'+(' --resume' if phase in ('train', 'evaluate') else '') for phase in ('prepare', 'banks', 'pilot', 'train', 'decide', 'evaluate')]
    lines += ['.venv-pytorch/bin/python scripts/report_m3w_european_producer_conditioned.py', '```', '',
        'For an existing completed directory, run the reporter for verification. Do not restart a pilot over a completed training checkpoint.',
        'Private source files, packed arrays and parent checkpoint banks are required to recreate predictions. Public aggregates alone do not reproduce private forecasts.',
        'A fixed final checkpoint was preregistered; no post-readout checkpoint/threshold selection. No reserved role was opened.']
    (run.PUBLIC/'execution_notes.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, subset in zip(axes, ('all', 'hard')):
        for i, c in enumerate(COMPARISONS):
            for h, color in ((0, '#087e8b'), (1, '#c74b36')):
                vals = [r['contrasts'][f'half{h}'][c]['ADE'][subset]['equal_scene_gain_percent'] for r in rows.values()]
                ax.scatter(np.full(len(vals), i+(h-.5)*.15), vals, color=color, alpha=.65, s=15, label=f'Bundle {h}' if i == 0 else None)
        ax.axhline(0, color='black', linewidth=.7); ax.set_xticks(range(4), COMPARISONS, rotation=12)
        ax.set_ylabel(f'{subset.capitalize()} ADE gain: producer minus control (%)'); ax.legend()
    fig.suptitle('Same forecasts, different controller inputs: all 36 dependent views')
    fig.savefig(run.PUBLIC/'paired_changes.svg'); fig.savefig(run.PRIVATE/'paired_changes.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for arm, color in zip(run.ARMS, ('#33434a', '#087e8b', '#c74b36')):
        for h in heads.values():
            if h['arm'] != arm: continue
            idx = 0 if h['task'] == 'utility' else 1
            ts = h['fit']['trace'] if idx == 0 else h['fit']['fixed_trace']
            axes[idx].plot([t['step'] for t in ts], [t['loss'] for t in ts], color=color, alpha=.22)
        for ax in axes: ax.plot([], [], color=color, label=arm)
    for ax, title in zip(axes, ('Utility: changing training batches', 'Risk: fixed training batch')):
        ax.set_title(title); ax.set_xlabel('Optimizer update'); ax.set_ylabel('Training loss'); ax.legend(); ax.set_yscale('log')
    fig.suptitle('All 108 heads, not validation-selected curves')
    fig.savefig(run.PUBLIC/'training_losses.svg'); fig.savefig(run.PRIVATE/'training_losses.png', dpi=160); plt.close(fig)
    for svg in run.PUBLIC.glob('*.svg'): svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    rows, heads, refs, counts, barrier, events = verify()
    summary = publish(rows, heads); documents(summary, heads, counts, barrier, rows)
    log = run.PRIVATE/'scoped_tests.log'; files = json.loads((run.PRIVATE/'scoped_test_files.json').read_text())
    match = re.search(r'(\d+) passed', log.read_text())
    if not match or ' failed' in log.read_text() or len(files) != 56: raise ValueError('Complete scoped test manifest required')
    phases = []
    for start in events:
        if start['state'] != 'phase_started': continue
        ends = [e for e in events if e['state'] == 'phase_complete' and e.get('phase') == start['phase'] and e['utc'] >= start['utc']]
        if not ends: raise ValueError('Unfinished phase')
        end = min(ends, key=lambda e: e['utc'])
        phases.append(dict(phase=start['phase'], pid=start['pid'], start=start['utc'], end=end['utc'],
            elapsed_seconds=(datetime.fromisoformat(end['utc'])-datetime.fromisoformat(start['utc'])).total_seconds()))
    q = json.loads((run.PRIVATE/'create_queue.json').read_text())
    pilot = json.loads((run.PRIVATE/'pilot.json').read_text())
    pilot_seconds = {r['identity']['task']: r['fit']['seconds'] for r in pilot['heads']}
    assert all(r['fit']['step'] == 100 for r in pilot['heads'])
    dump(run.PUBLIC/'compute_receipt.json', dict(phases=phases, runtime='native_arm64_cpu4_interop1_workers0',
        new_controller_heads=108, new_updates=216000, new_forecaster_training=False, local_resource_failure=False,
        pilot_optimizer_seconds=pilot_seconds, pilot_receipt_sha256=run.digest(run.PRIVATE/'pilot.json'),
        pilot_optimizer_only_full_budget_estimate_seconds=sum(pilot_seconds.values())*20*54,
        pilot_estimate_excludes_loading_inference_verification=True,
        live_resource_snapshot=json.loads((run.PRIVATE/'resource_snapshot.json').read_text()),
        create_queue=dict(successful_readonly_query=q['response']['returncode'] == 0,
            start=q['started_utc'], end=q['completed_utc'], jobs_submitted=q['jobs_submitted'], remote_modified=q['remote_modified'],
            private_receipt_sha256=run.digest(run.PRIVATE/'create_queue.json')),
        remote_artifact_store='not_inspected_authorized_M3W_path_unknown_no_absence_claim'))
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, checks=counts, group_receipts=refs,
        decision_barrier=barrier, new_checkpoint_prefix_replays=216, cached_neural_prefix_replays=18,
        cached_floor_prefix_replays=72, tests_passed=int(match.group(1)), scoped_test_files=files,
        test_source_hashes={f: run.digest(ROOT/f) for f in files}, test_log_sha256=run.digest(log),
        full_legacy_suite_run=False, reporter_sha256=run.digest(Path(__file__)),
        summary_sha256=run.digest(run.PUBLIC/'summary_metrics.json'),
        artifact_hashes={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'completion_checks.json'},
        new_neural_training=True, new_trajectory_training=False, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(status='complete', new_heads=108, updates=216000, **counts, tests=int(match.group(1)))), flush=True)


if __name__ == '__main__': main()
