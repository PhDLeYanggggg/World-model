"""Publish the entire registered incremental-policy comparison, not a winner."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_incumbent_relative as run
from scripts import report_m3w_european_fixed_producer_roles as previous
from scripts.report_m3w_european_floor_relative import spread, dump
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_causal_abstention import fmt

POLICIES = ('floor_reference', 'incumbent_reference', 'add_only', 'remove_only', 'ridge_incumbent', 'old_stop', 'previous_matched', 'raw_neural')
SUBSETS = ('all', 'easy', 'hard', 'complete')
CONTROLS = ('floor_reference', 'ridge_incumbent', 'old_stop', 'previous_matched')


def gates(summary):
    v = summary['policies']['incumbent_reference']; safety = v['safety']
    def positive_without_negative(m): return m['positive_CI'] > 0 and m['negative_CI'] == 0 and m['defined'] == 36
    easy = safety['worst_positive_easy_degradation_percent']
    return dict(training_complete=True, source_exclusion_verified=True, decisions_frozen_before_readout=True,
        incremental_gain_consistent=positive_without_negative(v['ADE_vs_old_stop4']['all']),
        matched_target_comparison_consistent=positive_without_negative(summary['incumbent_vs_controls']['floor_reference']['ADE']['all']),
        easy_preservation=easy is not None and easy <= 2 and safety['easy_defined_views'] == 36,
        zero_CV_harm_absent=safety['zero_CV_harm_views'] == 0,
        source_bootstrap_complete=True, independent_calibration=False, independent_confirmation=False,
        deployment_promoted=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def verify():
    run.ensure_frozen(); cfg = json.loads((ROOT/run.CONFIG).read_text())
    done = json.loads((run.PRIVATE/'evaluation_complete.json').read_text()); assert done['all_passed']
    identity = done['identity']; training = run.checked_training(identity)
    rows, heads = {}, {}
    counts = dict(saved_decisions_verified=0, coordinate_arrays_verified=0, metric_reductions_verified=0)
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text()); assert r['identity'] == identity and r['verified']
        assert set(r['views']) == set(POLICIES) and set(r['contrasts']) == set(CONTROLS)
        assert len({r[k] for k in ('producer', 'controller', 'readout')}) == 3
        for v in r['views'].values():
            assert v['ADE_vs_floor4']['all']['expected_scenes'] == identity['rosters'][r['readout']]
        rows[r['group']] = r
        for k in counts: counts[k] += r[k]
    assert len(rows) == 36 and counts == dict(saved_decisions_verified=288, coordinate_arrays_verified=144, metric_reductions_verified=3600)
    for ref in training['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); fit = r['fit']
        assert fit['complete'] and fit['step'] == 2000 and fit['total_draws'] == 512000 and fit['unknown_rows_sampled'] == 0
        assert r['replay_exact'] and r['prefix_replay_rows'] == 4096
        assert all(np.isfinite(t['loss']) and np.isfinite(t['gradient_norm']) for t in fit['trace'])
        heads[Path(ref['path']).parent.name] = dict(arm=r['identity']['arm'], task=r['identity']['task'],
            event=r['identity']['event'], seed=r['identity']['seed'], group=r['identity']['group'], fit=fit,
            checkpoint=r['artifacts']['checkpoint'], receipt_sha256=ref['sha256'],
            **{k: r['identity'][k] for k in ('label_sha256', 'feature_train_sha256', 'fitting_ids_sha256')})
    assert len(heads) == 144
    for ref in training['ridge']:
        assert json.loads((ROOT/ref['path']).read_text())['full_replay_exact']
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    barrier = {p: min(e['utc'] for e in events if e['state'] == 'phase_complete' and e.get('phase') == p)
        for p in ('prepare', 'pilot', 'train', 'decide')}
    barrier['first_new_readout'] = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    assert barrier['prepare'] < barrier['pilot'] < barrier['train'] < barrier['decide'] < barrier['first_new_readout']
    return cfg, rows, heads, counts, barrier, events


def publish(rows, heads):
    policies = {p: previous.summarize([r['views'][p] for r in rows.values()]) for p in POLICIES}
    for p in policies:
        policies[p]['override_rate_range'] = [min(r['views'][p]['override_rate'] for r in rows.values()), max(r['views'][p]['override_rate'] for r in rows.values())]
    strata = {}
    for a in range(3):
        for b in range(3):
            if a == b: continue
            for event in ('all', 'easy'):
                rr = [r for r in rows.values() if (r['producer'], r['controller'], r['event']) == (a, b, event)]
                assert len(rr) == 3
                strata[f'producer{a}_controller{b}_{event}'] = {p: previous.summarize([r['views'][p] for r in rr]) for p in POLICIES}
    summary = dict(result_source='fresh_run_144_Torch_heads_72_ridge_fits_288_frozen_readouts',
        forecasters='cached_verified', new_trajectory_training=False, heads=144, updates=288000, ridge_fits=72, views=288,
        policies=policies, incumbent_vs_controls={p: previous.old.paired_summary([r['contrasts'][p] for r in rows.values()]) for p in CONTROLS},
        role_and_event=strata, reliability={p: previous.old.reliability_summary([r['views'][p] for r in rows.values()])
            for p in ('floor_reference', 'incumbent_reference', 'ridge_incumbent')},
        bootstrap_resamples=3000, bootstrap_seed=39271, bootstrap_unit='source_locality',
        expected_localities_per_view=4, independent_confirmation=False, threshold_selection=False, deployment_changed=False)
    for group, r in rows.items():
        views = {}
        for p, v in r['views'].items():
            views[p] = dict(**{f: {s: compact_metric(m, scenes=True) for s, m in v[f].items()} for f in previous.FAMILIES},
                FDE_vs_floor4=compact_metric(v['FDE_vs_floor4'], scenes=True), easy_vs_CV=compact_metric(v['easy_vs_CV'], scenes=True),
                **{k: v[k] for k in ('switch_rate', 'override_rate', 'added_rows', 'removed_rows', 'switched_rows', 'unknown_ADE_switches', 'zero_CV')})
            if 'reliability' in v: views[p]['reliability'] = v['reliability']
        contrasts = {k: dict(ADE={s: compact_metric(m, scenes=True) for s, m in v['ADE'].items()},
            FDE=compact_metric(v['FDE'], scenes=True)) for k, v in r['contrasts'].items()}
        dump(run.PUBLIC/'groups'/(group+'.json'), dict(views=views, contrasts=contrasts,
            accounting=r['accounting'], **{k: r[k] for k in ('producer', 'controller', 'readout', 'seed', 'event')}))
    for name, h in heads.items(): dump(run.PUBLIC/'training'/(name+'.json'), h)
    dump(run.PUBLIC/'summary_metrics.json', summary)
    accounting = {}
    for group, r in rows.items():
        accounting[group] = {}
        for p, subsets in r['accounting'].items():
            accounting[group][p] = {}
            for subset, localities in subsets.items():
                terms = [v['percent_terms'] for v in localities.values() if v.get('percent_terms') is not None]
                if len(terms) != 4:
                    accounting[group][p][subset] = dict(status='incomplete_locality_support'); continue
                avg = {k: float(np.mean([v[k] for v in terms])) for k in terms[0]}
                net = avg['added_harm']-avg['added_benefit']+avg['removed_lost_benefit']-avg['removed_avoided_harm']
                m = r['views'][p]['ADE_vs_old_stop4'][subset]
                np.testing.assert_allclose(net, -m['equal_scene_gain_percent'], rtol=1e-9, atol=1e-9)
                accounting[group][p][subset] = dict(percent_terms=avg, net_degradation_percent=net, gain_ci=m['scene_bootstrap_ci95'])
    dump(run.PUBLIC/'changed_action_accounting.json', dict(status='posthoc_arithmetic_frozen_choices_not_new_policy', groups=accounting))
    return summary


def documents(s, heads, counts, barrier, rows):
    lines = ['# Incumbent-Relative Intervention: Complete Development Results', '',
        'I trained144 Torch heads and72 ridge controls on the same four-source forecasts and381 causal inputs.',
        'Only the learning reference and corresponding fallback rule change. All288 registered decisions froze before new readout.',
        'Three seeds, six ordered source rotations and two event targets are dependent development views, not independent tests.', '',
        '| Policy | All ADE gain vs incumbent (%) | Positive / negative CI | Hard gain vs incumbent (%) | Worst easy degradation vs CV (%) | Override rate |',
        '|---|---:|---:|---:|---:|---:|']
    for p, v in s['policies'].items():
        m = v['ADE_vs_old_stop4']['all']; safety = v['safety']
        lines.append(f"|{p}|{fmt(m['range'])}|{m['positive_CI']} / {m['negative_CI']}|{fmt(v['ADE_vs_old_stop4']['hard']['range'])}|{safety['worst_positive_easy_degradation_percent']:.6f}|{fmt(v['override_rate_range'])}|")
    lines += ['', '| Incumbent-relative vs | All gain (%) | Positive / negative CI | Hard gain (%) | Endpoint FDE gain (%) |', '|---|---:|---:|---:|---:|']
    for p, v in s['incumbent_vs_controls'].items():
        m = v['ADE']['all']; lines.append(f"|{p}|{fmt(m['range'])}|{m['positive_CI']} / {m['negative_CI']}|{fmt(v['ADE']['hard']['range'])}|{fmt(v['FDE']['range'])}|")
    lines += ['', '## Learned Risk Is Not Calibrated Safety', '',
        '| Head | Supported selected locality/views | Realized harm ratio >2% | Underpredicted | Realized ratio range |', '|---|---:|---:|---:|---:|']
    for p, r in s['reliability'].items():
        lines.append(f"|{p}|{r['selected_supported_views']}|{r['realized_above_2pct']}|{r['underpredicted_views']}|{fmt(r['realized_harm_ratio_range'])}|")
    lines += ['', 'For incumbent-relative heads, selected means an override, and the harm denominator is the incumbent cost; for floor heads it means selecting neural over the floor.',
        'These are different reference-cost ratios, not directly interchangeable safety certificates. Net easy degradation is a separate quantity.', '',
        'Every source/event/seed, partial-label and complete-window comparison, p95/p99 tail, endpoint FDE, zero-CV result and changed-action accounting is retained in groups/*.json.',
        'Bootstrap:3,000 paired source-locality resamples, four localities per view. Repeated windows are not IID. No multiplicity-adjusted confirmatory claim.',
        'Same feature statistics, supervision masks, source weights, draw counts, capacity, seeds and budget are verified. Target-dependent initial biases and ranking scales legitimately differ.',
        'No new trajectory forecaster training, independent calibration, reserved-role opening or deployment. Image pixels, obs8/pred12 rawstride12, released detector tracks; no metric/seconds/human-gold/physical-safety/true3D/foundation claim.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    gg = gates(s); dump(run.PUBLIC/'gates.json', gg)
    (run.PUBLIC/'gates.md').write_text('# Evidence Gates\n\n'+''.join(f'- {k}: {v}\n' for k, v in gg.items())+
        '\nFalse execution flags are prohibitions, not research passes. No automatic deployment promotion.\n')
    train = {}
    for arm in ('floor_reference', 'incumbent_reference'):
        train[arm] = {}
        for task in ('utility', 'risk'):
            hh = [h for h in heads.values() if (h['arm'], h['task']) == (arm, task)]
            fits = [h['fit'] for h in hh]
            t = dict(heads=len(fits), updates=sum(f['step'] for f in fits), parameters=sorted({f['parameters'] for f in fits}),
                unknown_rows_sampled=sum(f['unknown_rows_sampled'] for f in fits),
                distinct_input_target_bindings=len({(h['feature_train_sha256'], h['label_sha256'], h['fitting_ids_sha256']) for h in hh}))
            for label, index in [('first_sampled_loss', 0), ('final_sampled_loss', -1)]:
                vals = [f['trace'][index]['loss'] for f in fits]; t[label+'_range'] = [min(vals), max(vals)]
            if task == 'risk': t['fixed_training_batch_loss_reduced_heads'] = sum(f['fixed_trace'][-1]['loss'] < f['fixed_trace'][0]['loss'] for f in fits)
            train[arm][task] = t
    dump(run.PUBLIC/'training_summary.json', train)
    note = ['# Reproduction And Data Boundaries', '',
        'Registration2ebde873 was pushed before real fitting. Native arm64 CPU4/inter-op1/workers0.',
        'Pilot100 steps resumed inside the2,000-step budget;144 checkpoint4096-row prefix replays and72 complete ridge score replays.',
        'Cached forecasters retain prior verified lineage;144 incumbent-head inference calls and72 original-policy replays on B/C passed this round.',
        'Changing-batch loss is not validation. Fixed risk traces are training diagnostics, not validation evidence.',
        'No C fitting statistics; A easy/hard cutoffs, B supervised losses/normalization, C opened development readout only.',
        '', '```json', json.dumps(dict(checks=counts, barrier=barrier), indent=2), '```', '', '```sh']
    note += [f'.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase {p}'+(' --resume' if p in ('train', 'evaluate') else '') for p in ('prepare', 'pilot', 'train', 'decide', 'evaluate')]
    note += ['.venv-pytorch/bin/python scripts/report_m3w_european_incumbent_relative.py', '```', '',
        'For a completed run use the reporter. Resume only the identical interrupted run; do not restart the pilot over completed heads.',
        'Public code/config/aggregate receipts do not contain private raw tracks, forecast caches or checkpoints. Full training requires the verified local asset chain.',
        'The exact scoped test list and hashes are recorded in completion_checks.json. Full legacy suite is not_run.']
    (run.PUBLIC/'reproducibility.md').write_text('\n'.join(note)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, subset in zip(axes, ('all', 'hard')):
        for i, p in enumerate(CONTROLS):
            for fold, color in enumerate(('#087e8b', '#c74b36', '#525e70')):
                vals = [r['contrasts'][p]['ADE'][subset]['equal_scene_gain_percent'] for r in rows.values() if r['producer'] == fold]
                ax.scatter(np.full(len(vals), i+(fold-1)*.13), vals, color=color, alpha=.65, s=15, label=f'Producer fold{fold}' if i == 0 else None)
        ax.axhline(0, color='black', linewidth=.7); ax.set_xticks(range(4), ['floor ref', 'ridge', 'incumbent', 'prior matched'])
        ax.set_ylabel(f'{subset.capitalize()} ADE gain: incremental vs control (%)'); ax.legend()
    fig.suptitle('Same forecasts and inputs: all36 dependent role/seed/event views')
    fig.savefig(run.PUBLIC/'paired_changes.svg'); fig.savefig(run.PRIVATE/'paired_changes.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for arm, color in zip(('floor_reference', 'incumbent_reference'), ('#087e8b', '#c74b36')):
        for h in heads.values():
            if h['arm'] != arm: continue
            idx = int(h['task'] == 'risk'); trace = h['fit']['fixed_trace'] if idx else h['fit']['trace']
            axes[idx].plot([t['step'] for t in trace], [t['loss'] for t in trace], color=color, alpha=.15)
        for ax in axes: ax.plot([], [], color=color, label=arm)
    for ax, title in zip(axes, ('Utility: changing training batches', 'Risk: fixed training batch')):
        ax.set_title(title); ax.set_xlabel('Optimizer update'); ax.set_ylabel('Training loss'); ax.set_yscale('log'); ax.legend()
    fig.suptitle('All144 cost heads; loss curves are not validation or readout metrics')
    fig.savefig(run.PUBLIC/'training_losses.svg'); fig.savefig(run.PRIVATE/'training_losses.png', dpi=160); plt.close(fig)
    for svg in run.PUBLIC.glob('*.svg'): svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')


def main():
    cfg, rows, heads, counts, barrier, events = verify()
    s = publish(rows, heads); documents(s, heads, counts, barrier, rows)
    testfiles = json.loads((run.PRIVATE/'scoped_test_files.json').read_text())
    log = (run.PRIVATE/'scoped_tests.log').read_text(); match = re.search(r'(\d+) passed', log)
    if not match or ' failed' in log: raise ValueError('Scoped tests must pass before completion')
    phases = []
    for start in events:
        if start['state'] != 'phase_started': continue
        ends = [e for e in events if e['state'] == 'phase_complete' and e.get('phase') == start['phase'] and e['pid'] == start['pid']]
        if not ends: raise ValueError('Unfinished phase')
        end = min(ends, key=lambda e: e['utc'])
        phases.append(dict(phase=start['phase'], pid=start['pid'], start_utc=start['utc'], end_utc=end['utc'],
            seconds=int((datetime.fromisoformat(end['utc'])-datetime.fromisoformat(start['utc'])).total_seconds())))
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    dump(run.PUBLIC/'compute_receipt.json', dict(runtime='native_arm64_Torch_CPU4_interop1_workers0', phases=phases,
        create_readonly=dict(returncode=queue['response']['returncode'], receipt_sha256=run.digest(run.PRIVATE/'create_queue.json'),
            completed_utc=queue['completed_utc'], jobs_submitted=0, remote_modified=False),
        authorized_remote_artifact_directory='not_known_not_proven_absent', downgrade=False))
    hashes = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in sorted(run.PUBLIC.rglob('*')) if p.is_file() and p.name != 'completion_checks.json'}
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, artifact_hashes=hashes, counts=counts,
        source_bindings={p: run.digest(ROOT/p) for p in (*run.FILES, 'scripts/report_m3w_european_incumbent_relative.py')},
        test_files={p: run.digest(ROOT/p) for p in testfiles}, test_count=int(match[1]), scoped_test_files=len(testfiles),
        full_legacy_suite='not_run', barrier=barrier, evaluation_receipt=run.artifact(run.PRIVATE/'evaluation_complete.json'),
        training_receipt=run.artifact(run.PRIVATE/'training_complete.json')))
    print(json.dumps(dict(gates=gates(s), policies={p: v['ADE_vs_old_stop4']['all'] for p, v in s['policies'].items()})), flush=True)


if __name__ == '__main__': main()
