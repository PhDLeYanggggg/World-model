"""Report all registered fixed-producer directions, including negative readouts."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_fixed_producer_roles as run
from scripts import report_m3w_european_producer_conditioned as old
from scripts.report_m3w_european_floor_relative import spread, dump
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_causal_abstention import fmt

POLICIES = ('producer_matched', 'oof_control', 'ridge', 'old_stop', 'raw_neural')
SUBSETS = ('all', 'easy', 'hard', 'complete')
COMPARISONS = ('oof_control', 'ridge', 'old_stop')
FAMILIES = ('ADE_vs_floor4', 'ADE_vs_old_stop4')
EXPECTED = dict(saved_decisions_verified=180, coordinate_arrays_verified=144, metric_reductions_verified=2376)


def verify():
    run.ensure_frozen(); cfg, _, _, _, identity = run.load()
    done = json.loads((run.PRIVATE/'evaluation_complete.json').read_text())
    assert done['identity'] == identity and done['all_passed'] and len(done['groups']) == 36
    rows, counts = {}, dict.fromkeys(EXPECTED, 0)
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text())
        assert r['identity'] == identity and r['verified'] and set(r['views']) == set(POLICIES)
        assert set(r['contrasts']) == set(COMPARISONS)
        assert len({r['producer'], r['controller'], r['readout']}) == 3
        for v in r['views'].values():
            assert v['ADE_vs_floor4']['all']['expected_scenes'] == identity['rosters'][r['readout']]
            assert len(v['ADE_vs_floor4']['all']['by_scene']) == 4
        for k in counts: counts[k] += r[k]
        rows[r['group']] = r
    assert len(rows) == 36 and counts == EXPECTED
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    barrier = {p: min(e['utc'] for e in events if e['state'] == 'phase_complete' and e.get('phase') == p)
               for p in ('prepare', 'pilot', 'train', 'decide')}
    barrier['first_new_readout'] = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    assert barrier['prepare'] < barrier['pilot'] < barrier['train'] < barrier['decide'] < barrier['first_new_readout']
    training = run.checked_training(identity); heads = {}
    for ref in training['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); fit = r['fit']
        assert fit['complete'] and fit['step'] == 2000 and fit['total_draws'] == 512000
        assert fit['unknown_rows_sampled'] == 0 and r['prefix_replay_rows'] == 4096 and r['replay_exact']
        assert all(np.isfinite(t['loss']) and np.isfinite(t['gradient_norm']) for t in fit['trace'])
        heads[Path(ref['path']).parent.name] = dict(arm=r['identity']['arm'], task=r['identity']['task'], event=r['identity']['event'],
            seed=r['identity']['seed'], group=r['identity']['group'], fit=fit,
            checkpoint=r['artifacts']['checkpoint'], receipt_sha256=ref['sha256'],
            label_sha256=r['identity']['label_sha256'], feature_train_sha256=r['identity']['feature_train_sha256'],
            fitting_ids_sha256=r['identity']['fitting_ids_sha256'])
    assert len(heads) == cfg['new_neural_heads'] == 144
    for ref in training['ridge']:
        r = json.loads((ROOT/ref['path']).read_text()); assert r['full_replay_exact']
    return rows, heads, done['groups'], counts, barrier, events


def easy_decomposition(view):
    converted = dict(view, ADE_vs_floor2=view['ADE_vs_floor4'])
    result = old.easy_decomposition(converted)
    for r in result.values():
        if r['supported']: r['floor4_degradation_vs_CV'] = r.pop('floor2_degradation_vs_CV')
    return result


def summarize(views):
    safety = old.safety(views)
    safety['easy_defined_views'] = sum(v['easy_vs_CV']['worst_scene_gain_percent'] is not None for v in views)
    safety['easy_expected_views'] = len(views)
    return dict(**{f: {s: spread([v[f][s] for v in views]) for s in SUBSETS} for f in FAMILIES},
        FDE_vs_floor4=spread([v['FDE_vs_floor4'] for v in views]), easy_vs_CV=spread([v['easy_vs_CV'] for v in views]),
        safety=safety)


def no_promotion_gates(summary):
    """Evidence conditions, not a post-readout model-selection rule."""
    m = summary['policies']['producer_matched']; easy = m['safety']['worst_positive_easy_degradation_percent']
    contrasts = summary['matched_vs_controls']
    return dict(data_and_lineage=True, all_registered_training_complete=True, frozen_before_readout=True,
        matched_gain_consistent_across_controls=all(v['ADE']['all']['negative_CI'] == 0 and v['ADE']['all']['positive_CI'] > 0 for v in contrasts.values()),
        matched_easy_preserved=easy is not None and easy <= 2. and m['safety']['easy_defined_views'] == m['safety']['easy_expected_views'] == 36,
        observed_zero_CV_harm_absent=m['safety']['zero_CV_harm_views'] == 0,
        source_bootstrap_complete=True, independent_confirmation=False, deployment_promoted=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def publish(rows, heads):
    decomposition = {}
    for group, r in rows.items():
        decomposition[group] = {p: easy_decomposition(v) for p, v in r['views'].items()}
        compact = {p: dict(**{f: {s: compact_metric(m) for s, m in v[f].items()} for f in FAMILIES},
            FDE_vs_floor4=compact_metric(v['FDE_vs_floor4']), easy_vs_CV=compact_metric(v['easy_vs_CV']),
            **{k: v[k] for k in ('switch_rate', 'switched_rows', 'unknown_ADE_switches', 'zero_CV', 'reliability')}) for p, v in r['views'].items()}
        contrasts = {k: dict(ADE={s: compact_metric(m) for s, m in c['ADE'].items()}, FDE=compact_metric(c['FDE'])) for k, c in r['contrasts'].items()}
        dump(run.PUBLIC/'groups'/(group+'.json'), dict(views=compact, contrasts=contrasts,
            floor_easy_vs_CV=compact_metric(r['floor_easy_vs_CV'], scenes=True),
            producer=r['producer'], controller=r['controller'], readout=r['readout'], seed=r['seed'], event=r['event']))
        with (run.PUBLIC/'groups'/(group+'_localities.csv')).open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99', 'reference_p95')
            writer.writerow(['policy', 'comparison', 'subset', 'locality', *fields])
            for p, v in r['views'].items():
                ms = [(fam, s, m) for fam in FAMILIES for s, m in v[fam].items()]
                ms += [('FDE_vs_floor4', 'endpoint', v['FDE_vs_floor4']), ('ADE_vs_CV', 'easy', v['easy_vs_CV'])]
                for fam, s, m in ms:
                    for site, rr in m['by_scene'].items(): writer.writerow([p, fam, s, site, *[rr.get(k) for k in fields]])
            for other, c in r['contrasts'].items():
                for s, m in {**c['ADE'], 'endpoint_FDE': c['FDE']}.items():
                    for site, rr in m['by_scene'].items(): writer.writerow(['producer_matched', other, s, site, *[rr.get(k) for k in fields]])
    pooled = {p: summarize([r['views'][p] for r in rows.values()]) for p in POLICIES}
    paired = {c: old.paired_summary([r['contrasts'][c] for r in rows.values()]) for c in COMPARISONS}
    strata = {}
    for a in range(3):
        for b in range(3):
            if a == b: continue
            for event in ('all', 'easy'):
                rr = [r for r in rows.values() if (r['producer'], r['controller'], r['event']) == (a, b, event)]
                assert len(rr) == 3
                strata[f'producer{a}_controller{b}_{event}'] = dict(
                    policies={p: summarize([r['views'][p] for r in rr]) for p in POLICIES},
                    contrasts={c: old.paired_summary([r['contrasts'][c] for r in rr]) for c in COMPARISONS})
    attribution = {}
    for p in POLICIES:
        rr = [r for g in decomposition.values() for r in g[p].values() if r['supported']]
        attribution[p] = dict(dependent_locality_views=len(rr),
            **{k: sum(r[k] for r in rr) for k in ('already_bad_floor', 'controller_created_violation', 'violation_despite_controller_improvement')},
            total_violations=sum(r['total_degradation_vs_CV'] > 2. for r in rr),
            floor_degradation_range=[min(r['floor4_degradation_vs_CV'] for r in rr), max(r['floor4_degradation_vs_CV'] for r in rr)],
            controller_added_degradation_range_pp=[min(r['controller_added_degradation_pp'] for r in rr), max(r['controller_added_degradation_pp'] for r in rr)])
    summary = dict(result_source='fresh_run_144_neural_heads_72_ridge_fits_180_readouts', trajectory_forecasters='cached_verified',
        new_controller_training=True, new_trajectory_training=False, heads=144, updates=288000, ridge_fits=72, views=180,
        bootstrap_resamples=3000, bootstrap_seed=39271, bootstrap_unit='source_locality', expected_localities_per_view=4,
        policies=pooled, matched_vs_controls=paired, role_and_event=strata,
        reliability={p: old.reliability_summary([r['views'][p] for r in rows.values()]) for p in POLICIES[:-1]},
        posthoc_easy_attribution=attribution, independent_confirmation=False, threshold_selection=False, deployment_changed=False)
    dump(run.PUBLIC/'easy_attribution.json', dict(status='posthoc_arithmetic_not_new_policy', groups=decomposition, summary=attribution))
    dump(run.PUBLIC/'summary_metrics.json', summary)
    for name, h in heads.items(): dump(run.PUBLIC/'training'/(name+'.json'), h)
    return summary


def training_summary(heads):
    result = {}
    for arm in POLICIES[:2]:
        result[arm] = {}
        for task in ('utility', 'risk'):
            fits = [r['fit'] for r in heads.values() if (r['arm'], r['task']) == (arm, task)]
            first, final = [f['trace'][0]['loss'] for f in fits], [f['trace'][-1]['loss'] for f in fits]
            item = dict(heads=len(fits), updates=sum(f['step'] for f in fits), parameters=sorted(set(f['parameters'] for f in fits)),
                sampled_first_loss_range=[min(first), max(first)], sampled_final_loss_range=[min(final), max(final)],
                unknown_rows_sampled=sum(f['unknown_rows_sampled'] for f in fits), optimizer_seconds_sum=sum(f['seconds'] for f in fits))
            item['distinct_fitting_input_target_bindings'] = len({(r['fitting_ids_sha256'], r['feature_train_sha256'], r['label_sha256'])
                for r in heads.values() if (r['arm'], r['task']) == (arm, task)})
            if task == 'risk':
                fixed = [(f['fixed_trace'][0]['loss'], f['fixed_trace'][-1]['loss']) for f in fits]
                item.update(fixed_training_batch_loss_reduced_heads=sum(b < a for a, b in fixed),
                    fixed_training_batch_final_loss_range=[min(b for _, b in fixed), max(b for _, b in fixed)])
            result[arm][task] = item
    return result


def documents(summary, heads, counts, barrier, rows):
    lines = ['# Fixed Four-Source Producer Roles: Complete Development Results', '',
        'I held the four-source candidate and protected fallback fixed and separated producer fitting A, controller supervision B, and readout C.',
        'Six ordered source rotations, three seeds and two event targets give36 dependent groups. All144 Torch heads reached2,000 updates;72 fixed-alpha ridge heads were also fitted.',
        'All180 decisions froze before new comparison readout. No trajectory forecaster was retrained. Reserved model-selection/calibration/confirmation roles remain closed.', '',
        '## Same-Forecast Primary Comparisons', '',
        '| Matched supervision vs | All ADE gain (%) | Positive / negative CI | Hard gain (%) | Complete gain (%) | Endpoint FDE gain (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for c, v in summary['matched_vs_controls'].items():
        m = v['ADE']['all']; lines.append(f"| {c} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {fmt(v['ADE']['hard']['range'])} | {fmt(v['ADE']['complete']['range'])} | {fmt(v['FDE']['range'])} |")
    lines += ['', '## Accuracy And Safety', '',
        '| Policy | All gain vs floor4 (%) | All gain vs old stop (%) | Positive / negative CI vs old stop | Worst easy degradation vs CV (%) | Zero-CV harmed views | Switch rate |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p, v in summary['policies'].items():
        m, s = v['ADE_vs_old_stop4']['all'], v['safety']
        lines.append(f"| {p} | {fmt(v['ADE_vs_floor4']['all']['range'])} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {s['worst_positive_easy_degradation_percent']:.5f} | {s['zero_CV_harm_views']} | {fmt(s['switch_rate_range'])} |")
    lines += ['', 'Raw neural is unprotected diagnostic, not a candidate. Positive-harm ratios below are different from net easy degradation.', '',
        '| Policy | Supported selected locality/views | Above2% realized harm ratio | Underpredicted | Realized ratio range | Predicted ratio range |', '|---|---:|---:|---:|---:|---:|']
    for p, r in summary['reliability'].items():
        lines.append(f"| {p} | {r['selected_supported_views']} | {r['realized_above_2pct']} | {r['underpredicted_views']} | {fmt(r['realized_harm_ratio_range'])} | {fmt(r['predicted_harm_ratio_range'])} |")
    lines += ['', '## Easy Failure Accounting', '', '| Policy | Violations | Already-bad floor views | New violations from controller | Violations despite helpful increment |', '|---|---:|---:|---:|---:|']
    for p, r in summary['posthoc_easy_attribution'].items():
        lines.append(f"| {p} | {r['total_violations']} | {r['already_bad_floor']} | {r['controller_created_violation']} | {r['violation_despite_controller_improvement']} |")
    lines += ['', 'Every rotation, event, seed and source locality is retained in groups/*.json and *_localities.csv, including p95/p99 tails.',
        'Four-locality bootstraps use3,000 paired resamples per view, not independent overlapping windows. Views share source data and are not independent tests; no multiplicity-adjusted confirmatory claim is made.',
        'The matched/OOF contrast changes training producer, source count, rollout quality, labels and fitted feature statistics jointly. It is not isolated producer-identity causality.',
        'Both arms share supervised draw arrays, masks, source weights, CV-cost scales, capacity and budget. Their target labels, feature statistics and fixed ranking scales legitimately differ.',
        'The original A-derived easy/hard cutoffs are common. B supplies all new head supervision/normalization; C supplies no fitted statistics.',
        'Image-pixel obs8/pred12 at raw annotation stride12, EuropeanSquares released detector tracks. No metric/seconds, human gold, physical safety, true3D or foundation claim.',
        'Partial-window ADE and actual endpoint FDE are separate. Unknown future labels are not zero errors. No deployment change, Stage5C or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    train = training_summary(heads); dump(run.PUBLIC/'training_summary.json', train)
    lines = ['# Training And Reproduction', '',
        'Registration8dd1dd2d was pushed before the pilot and full fit. Native arm64 CPU4/inter-op1/workers0;380 inputs and width64.',
        'The100-update utility/risk pilot resumed inside each2,000-update budget.144 checkpoint prefixes replay exactly;72 ridge score arrays replay fully.',
        'All9 cached neural prefixes,72 old-head replays and36 old decisions passed before fitting. The preflight index error is documented, not hidden.',
        'Changing-batch losses are not validation performance. Risk also has a fixed training-batch trace. No readout-selected checkpoint or threshold.', '',
        '| Arm / head | Heads | Updates | First sampled loss | Last sampled loss | Fixed training batch decrease |', '|---|---:|---:|---:|---:|---:|']
    for arm, tasks in train.items():
        for task, t in tasks.items():
            lines.append(f"| {arm} / {task} | {t['heads']} | {t['updates']} | {fmt(t['sampled_first_loss_range'])} | {fmt(t['sampled_final_loss_range'])} | {t.get('fixed_training_batch_loss_reduced_heads', 'not measured')} |")
    lines += ['', 'Some OOF utility supervision repeats across producer rotations; these fresh paired refits are not independent replications. training_summary.json reports distinct fitting input/target bindings.',
        '', '```json', json.dumps(dict(checks=counts, barrier=barrier), indent=2), '```', '', '```sh']
    lines += [f'.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase {p}'+(' --resume' if p in ('train', 'evaluate') else '') for p in ('prepare', 'pilot', 'train', 'decide', 'evaluate')]
    lines += ['.venv-pytorch/bin/python scripts/report_m3w_european_fixed_producer_roles.py', '```', '',
        'For an existing completed experiment use the reporter, not another pilot. Active fits can resume with the same configuration and bindings.',
        'Public code/config/aggregate evidence is available; reproducing private forecasts requires the local source/cache/checkpoint chain. No raw tracks or checkpoints are committed.',
        'Full legacy test suite is not run; the exact scoped suite and source hashes appear in completion_checks.json.']
    (run.PUBLIC/'execution_notes.md').write_text('\n'.join(lines)+'\n')
    gates = no_promotion_gates(summary); dump(run.PUBLIC/'gates.json', gates)
    (run.PUBLIC/'gates.md').write_text('# Evidence Gates\n\n'+''.join(f'- {k}: {v}\n' for k, v in gates.items())+
        '\nImplementation checks and empirical effects are separate; false execution flags are prohibitions, not research passes. No deployment promotion or independent confirmation.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, subset in zip(axes, ('all', 'hard')):
        for i, c in enumerate(COMPARISONS):
            for fold, color in enumerate(('#087e8b', '#c74b36', '#525e70')):
                vals = [r['contrasts'][c]['ADE'][subset]['equal_scene_gain_percent'] for r in rows.values() if r['producer'] == fold]
                ax.scatter(np.full(len(vals), i+(fold-1)*.13), vals, color=color, alpha=.65, s=15, label=f'Producer fold{fold}' if i == 0 else None)
        ax.axhline(0, color='black', linewidth=.7); ax.set_xticks(range(3), COMPARISONS)
        ax.set_ylabel(f'{subset.capitalize()} ADE gain: matched vs control (%)'); ax.legend()
    fig.suptitle('Identical final forecasts: all 36 dependent role/seed/event views')
    fig.savefig(run.PUBLIC/'paired_changes.svg'); fig.savefig(run.PRIVATE/'paired_changes.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for arm, color in zip(POLICIES[:2], ('#087e8b', '#c74b36')):
        for h in heads.values():
            if h['arm'] != arm: continue
            idx = int(h['task'] == 'risk'); trace = h['fit']['fixed_trace'] if idx else h['fit']['trace']
            axes[idx].plot([t['step'] for t in trace], [t['loss'] for t in trace], color=color, alpha=.15)
        for ax in axes: ax.plot([], [], color=color, label=arm)
    for ax, title in zip(axes, ('Utility: changing training batches', 'Risk: fixed training batch')):
        ax.set_title(title); ax.set_xlabel('Optimizer update'); ax.set_ylabel('Training loss'); ax.set_yscale('log'); ax.legend()
    fig.suptitle('All 144 new heads; no validation or readout-selected curves')
    fig.savefig(run.PUBLIC/'training_losses.svg'); fig.savefig(run.PRIVATE/'training_losses.png', dpi=160); plt.close(fig)
    for svg in run.PUBLIC.glob('*.svg'): svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')


def main():
    rows, heads, refs, counts, barrier, events = verify()
    summary = publish(rows, heads); documents(summary, heads, counts, barrier, rows)
    log = run.PRIVATE/'scoped_tests.log'; files = json.loads((run.PRIVATE/'scoped_test_files.json').read_text())
    match = re.search(r'(\d+) passed', log.read_text())
    if not match or ' failed' in log.read_text() or len(files) != 60: raise ValueError('Complete scoped test manifest required')
    phases = []
    for start in events:
        if start['state'] != 'phase_started': continue
        ends = [e for e in events if e['state'] == 'phase_complete' and e.get('phase') == start['phase'] and e['pid'] == start['pid']]
        if not ends:
            if start['pid'] != 80496 or start['phase'] != 'prepare': raise ValueError('Unfinished phase')
            phases.append(dict(phase='prepare', pid=80496, status='pretraining_replay_error_repaired')); continue
        end = min(ends, key=lambda e: e['utc'])
        phases.append(dict(phase=start['phase'], pid=start['pid'], start=start['utc'], end=end['utc'],
            elapsed_seconds=(datetime.fromisoformat(end['utc'])-datetime.fromisoformat(start['utc'])).total_seconds()))
    q = json.loads((run.PRIVATE/'create_queue.json').read_text()); pilot = json.loads((run.PRIVATE/'pilot.json').read_text())
    accounting = json.loads((run.PUBLIC/'changed_action_accounting.json').read_text())
    assert set(accounting['groups']) == set(rows)
    assert accounting['evaluation_receipt'] == run.artifact(run.PRIVATE/'evaluation_complete.json')
    assert all(set(g) == {'all', 'easy', 'hard'} and all(len(v['localities']) == 4 for v in g.values()) for g in accounting['groups'].values())
    assert all(r['fit']['step'] == 100 for r in pilot['heads'])
    ptime = {r['identity']['task']: r['fit']['seconds'] for r in pilot['heads']}
    dump(run.PUBLIC/'compute_receipt.json', dict(phases=phases, runtime='native_arm64_cpu4_interop1_workers0',
        heads=144, updates=288000, ridge_fits=72, new_forecaster_training=False, local_resource_failure=False,
        pretraining_replay_error=True, pilot_seconds=ptime, pilot_optimizer_only_estimate_seconds=sum(ptime.values())*20*72,
        estimate_excludes_loading_ridge_inference_verification=True,
        resource_snapshot=json.loads((run.PRIVATE/'resource_snapshot.json').read_text()),
        create_queue=dict(successful_readonly_query=q['response']['returncode'] == 0,
            start=q['started_utc'], end=q['completed_utc'], jobs_submitted=q['jobs_submitted'], remote_modified=q['remote_modified'],
            private_receipt_sha256=run.digest(run.PRIVATE/'create_queue.json')),
        remote_artifact_store='not_inspected_authorized_M3W_path_unknown_no_absence_claim'))
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, checks=counts, group_receipts=refs,
        asset_receipt=run.artifact(run.PRIVATE/'assets_complete.json'),
        frozen_decision_receipt=run.artifact(run.PRIVATE/'decisions_complete.json'),
        trained_head_receipt=run.artifact(run.PRIVATE/'training_complete.json'),
        decision_barrier=barrier, new_checkpoint_prefix_replays=144, full_ridge_replays=72,
        posthoc_action_accounting_groups=36, posthoc_action_locality_tables=432,
        cached_neural_prefix_replays=9, cached_old_head_replays=72, old_choices_exact=36,
        tests_passed=int(match.group(1)), scoped_test_files=files, test_source_hashes={f: run.digest(ROOT/f) for f in files},
        test_log_sha256=run.digest(log), full_legacy_suite_run=False, reporter_sha256=run.digest(Path(__file__)),
        changed_action_accounting_script_sha256=run.digest(ROOT/'scripts/diagnose_m3w_european_fixed_producer_roles.py'),
        summary_sha256=run.digest(run.PUBLIC/'summary_metrics.json'),
        artifact_hashes={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'completion_checks.json'},
        new_neural_training=True, new_trajectory_training=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(status='complete', heads=144, updates=288000, **counts, tests=int(match.group(1)))), flush=True)


if __name__ == '__main__': main()
