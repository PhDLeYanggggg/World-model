"""Publish the complete registered abstention matrix without choosing a winner."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_causal_abstention as run
from scripts.report_m3w_european_floor_relative import spread, dump, verify as verify_parent
from scripts.report_m3w_european_protected_motion import compact_metric

POLICIES = ('original', 'stop', 'stop_risk', 'stop_random', 'support', 'support_risk', 'support_random',
            'combined', 'combined_risk', 'combined_random')


def verify():
    run.ensure_both_frozen(); verify_parent()
    counts = dict(saved_decisions_verified=0, independent_coordinate_arrays=0,
                  independent_metric_reductions=0, original_metrics_exact=0)
    rows, refs = {}, {}
    expected = {p+'__'+v for p in ('cv_targets', 'floor_both') for v in POLICIES}
    for mode in ('batch', 'fitting'):
        identity = json.loads((run.PRIVATE/mode/'identity.json').read_text())
        done = json.loads((run.PRIVATE/mode/'evaluation_complete.json').read_text())
        assert done['identity'] == identity and done['all_passed'] and len(done['groups']) == 18
        rows[mode] = {}; refs[mode] = done['groups']
        for ref in done['groups']:
            if run.artifact(ROOT/ref['path']) != ref: raise ValueError('Changed result')
            r = json.loads((ROOT/ref['path']).read_text())
            assert r['verified'] and r['identity'] == identity and set(r['views']) == expected
            for k in counts: counts[k] += r[k]
            for v in r['views'].values():
                assert len(v['ADE_vs_floor']['all']['expected_scenes']) == 8
            rows[mode][r['group']] = r
        assert len(rows[mode]) == 18
    assert counts == dict(saved_decisions_verified=720, independent_coordinate_arrays=144,
                          independent_metric_reductions=8208, original_metrics_exact=432)
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    first = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    frozen = {mode: min(e['utc'] for e in events if e['state'] == 'phase_complete' and e.get('phase') == 'decide'
                         and e['mode'] == mode) for mode in rows}
    assert all(t < first for t in frozen.values())
    return rows, refs, counts, dict(decisions_completed=frozen, first_new_readout=first)


def removal_summary(views):
    out = {}
    for subset in ('all', 'easy', 'hard', 'complete'):
        summaries = []
        for v in views:
            ledger = v['removal'][subset]
            records = [r for r in ledger.values() if r['floor_error_sum'] > 0]
            avoided = np.mean([100*r['avoided_harm']/r['floor_error_sum'] for r in records])
            lost = np.mean([100*r['lost_benefit']/r['floor_error_sum'] for r in records])
            summaries.append(dict(avoided_harm_pp=float(avoided), lost_benefit_pp=float(lost), change_pp=float(avoided-lost)))
        out[subset] = {k: [min(r[k] for r in summaries), max(r[k] for r in summaries)] for k in summaries[0]}
    return out


def summarize(views):
    out = {fam: {s: spread([v[fam][s] for v in views]) for s in views[0][fam]}
           for fam in ('ADE_vs_floor', 'ADE_vs_original')}
    out.update(FDE_vs_floor=spread([v['FDE_vs_floor'] for v in views]),
        easy_vs_CV=spread([v['ADE_vs_CV_easy'] for v in views]),
        zero_CV_harm_views=sum(v['zero_CV']['harmed_rows'] > 0 for v in views),
        zero_CV_supported_views=sum(v['zero_CV']['rows'] > 0 for v in views),
        switch_rate_range=[min(v['switch_rate'] for v in views), max(v['switch_rate'] for v in views)],
        unknown_ADE_switch_range=[min(v['unknown_ADE_switches'] for v in views), max(v['unknown_ADE_switches'] for v in views)],
        stopped_switch_range=[min(v['stopped_switches'] for v in views), max(v['stopped_switches'] for v in views)],
        removed_cost=removal_summary(views))
    if 'matched_controls' in views[0]:
        out['matched_controls'] = {c: {s: spread([v['matched_controls'][c][s] for v in views])
            for s in ('all', 'easy', 'hard', 'complete')} for c in ('risk', 'random')}
    return out


def publish(rows):
    summary = {}
    for mode, groups in rows.items():
        summary[mode] = {event: {key: summarize([r['views'][key] for group, r in groups.items() if group.endswith('_'+event)])
                                 for key in next(iter(groups.values()))['views']} for event in ('all', 'easy')}
        for group, r in groups.items():
            compact = {}
            for key, v in r['views'].items():
                compact[key] = {family: {s: compact_metric(m) for s, m in v[family].items()}
                                for family in ('ADE_vs_floor', 'ADE_vs_original')}
                compact[key].update(FDE_vs_floor=compact_metric(v['FDE_vs_floor']),
                    ADE_vs_CV_easy=compact_metric(v['ADE_vs_CV_easy']),
                    **{k: v[k] for k in ('switch_rate', 'switched_rows', 'unknown_ADE_switches', 'stopped_switches', 'zero_CV', 'removal')})
                if 'matched_controls' in v:
                    compact[key]['matched_controls'] = {c: {s: compact_metric(m) for s, m in metrics.items()}
                                                        for c, metrics in v['matched_controls'].items()}
            dump(run.PUBLIC/mode/(group+'.json'), dict(views=compact, queries=r['queries']))
            with (run.PUBLIC/mode/(group+'_localities.csv')).open('w', newline='') as f:
                w = csv.writer(f, lineterminator='\n')
                fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99', 'reference_p95')
                w.writerow(['policy', 'comparison', 'subset', 'locality', *fields])
                for key, v in r['views'].items():
                    metrics = [('ADE_vs_floor', s, m) for s, m in v['ADE_vs_floor'].items()]
                    metrics += [('ADE_vs_original', s, m) for s, m in v['ADE_vs_original'].items()]
                    metrics += [('FDE_vs_floor', 'endpoint', v['FDE_vs_floor']), ('ADE_vs_CV', 'easy', v['ADE_vs_CV_easy'])]
                    if 'matched_controls' in v:
                        metrics += [('matched_'+c, s, m) for c, ss in v['matched_controls'].items() for s, m in ss.items()]
                    for family, subset, m in metrics:
                        for site, row in m['by_scene'].items():
                            w.writerow([key, family, subset, site, *[row.get(k) for k in fields]])
    pooled = {key: summarize([r['views'][key] for groups in rows.values() for r in groups.values()])
              for key in next(iter(rows['batch'].values()))['views']}
    result = dict(result_source='fresh_run_guard_construction_and_evaluation', prior_forecasts_heads_population='cached_verified',
        new_neural_training=False, view_count=720, conditional_development_only=True,
        bootstrap_resamples=3000, bootstrap_unit='source_locality', bootstrap_seed=39271,
        modes=summary, across_dependent_views=pooled, deployment_changed=False, independent_confirmation=False)
    dump(run.PUBLIC/'summary_metrics.json', result)
    return result


def fmt(values):
    if values is None:
        return 'undefined'
    if max(abs(v) for v in values) < .001:
        return f'{values[0]:.6g} to {values[1]:.6g}'
    return f'{values[0]:.4f} to {values[1]:.4f}'


def decision_context(rows):
    """Label-free accounting of how many matched queries admit an actual choice."""
    out = {}
    for mode, groups in rows.items():
        out[mode] = {}
        for group in groups:
            path = run.PRIVATE/mode/'decisions'/(group+'.npz')
            receipt = json.loads(path.with_suffix('.json').read_text())
            with np.load(path, allow_pickle=False) as z:
                q = z['query']; nq = int(q.max())+1
                record = dict(fitting_boxes_by_state={str(s): sum(b['state'] == s for b in receipt['fitted_support']['boxes'])
                                                      for s in range(3)}, policies={})
                for parent in ('cv_targets', 'floor_both'):
                    original = z[parent+'__original']; original_count = np.bincount(q[original], minlength=nq)
                    for guard in ('stop', 'support', 'combined'):
                        key = parent+'__'+guard; chosen = z[key]; count = np.bincount(q[chosen], minlength=nq)
                        row = dict(indexed_queries=nq, original_active_queries=int((original_count > 0).sum()),
                            free_choice_queries=int(((count > 0) & (count < original_count)).sum()),
                            fully_rejected_queries=int(((count == 0) & (original_count > 0)).sum()))
                        for control in ('risk', 'random'):
                            other = z[key+'_'+control]
                            np.testing.assert_array_equal(count, np.bincount(q[other], minlength=nq))
                            row[control+'_different_rows'] = int((chosen != other).sum())
                            row[control+'_different_queries'] = len(set(q[chosen != other].tolist()))
                        record['policies'][key] = row
                out[mode][group] = record
    dump(run.PUBLIC/'decision_context.json', dict(result_source='fresh_label_free_accounting_on_frozen_decisions', modes=out,
        purpose='distinguish_count_matching_from_nontrivial_joint_allocation', posthoc_description_not_policy_selection=True))
    return out


def documents(summary, counts, barrier, rows):
    pooled = summary['across_dependent_views']
    lines = ['# Causal Abstention Results', '', 'Fresh causal guards and readout; cached_verified forecasters, heads and population.',
        'No neural training in this experiment. All 720 registered views evaluated, without selecting a deployment.', '',
        'Ranges below span 36 dependent development views per policy. Positive/negative CI counts',
        'are descriptive, not independent hypothesis tests or multiplicity-corrected evidence.', '',
        '| Parent / policy | All-ADE gain vs floor (%) | Gain vs original (%) | Positive / negative CI vs original | Easy worst locality gain vs CV (%) | Zero-CV harm views |',
        '|---|---:|---:|---:|---:|---:|']
    for key, v in pooled.items():
        m = v['ADE_vs_original']['all']
        lines.append(f"| {key} | {fmt(v['ADE_vs_floor']['all']['range'])} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {v['easy_vs_CV']['worst_locality']:.4f} | {v['zero_CV_harm_views']} / {v['zero_CV_supported_views']} supported |")
    lines += ['', '## Same-Frame Count-Matched Evidence', '',
        '| Policy | Control | All gain range (%) | Positive / negative CI | Complete-label gain range (%) | Hard gain range (%) |',
        '|---|---|---:|---:|---:|---:|']
    for key, v in pooled.items():
        for control, mm in v.get('matched_controls', {}).items():
            m = mm['all']
            lines.append(f"| {key} | {control} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {fmt(mm['complete']['range'])} | {fmt(mm['hard']['range'])} |")
    lines += ['', '## Lost Benefit and Avoided Harm', '',
        'Both quantities are divided by the same locality-specific floor-error sum, then averaged over the fixed roster.',
        'Their difference is a percentage-point change in gain over the floor, not a relative percentage vs the parent.', '',
        '| Policy | Avoided all harm (pp) | Lost all benefit (pp) | Net all change (pp) | Net hard change (pp) | Switch-rate range |',
        '|---|---:|---:|---:|---:|---:|']
    for key, v in pooled.items():
        r = v['removed_cost']
        lines.append(f"| {key} | {fmt(r['all']['avoided_harm_pp'])} | {fmt(r['all']['lost_benefit_pp'])} | {fmt(r['all']['change_pp'])} | {fmt(r['hard']['change_pp'])} | {fmt(v['switch_rate_range'])} |")
    lines += ['', 'Complete per-mode, fold, seed, event, subset and locality results are in the adjacent JSON/CSV files.',
        'CSV includes p95/p99 and worst-locality errors; unknown outcomes are never scored as zero.',
        'Query cohorts are the indexed agent histories at a current frame, not all visible agents.',
        'The comparison is not an interaction-conflict metric or a physical-safety certificate.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    checks = ['# Execution and Reproducibility', '', 'Registration commit: `3ff17937`, pushed before decision construction.',
        'Native arm64 Python; CPU four threads, inter-op one, no loader multiprocessing.',
        'No new neural weights or trajectory forecasters fitted; support boxes and decisions were freshly built.',
        '', '```json', json.dumps(dict(checks=counts, barrier=barrier), indent=2), '```', '',
        'The verifier hash-checks the prior 234-head experiment, both new decision banks, source exclusion,',
        'and all 36 complete result receipts. Scalar box/query ranking replay verifies all 720 saved policies.',
        'Only fitting rows contribute support boxes. No held labels determine guard decisions or frame counts.',
        'All eight-locality rosters retained. Risk-ranked and random controls match each guard within recording/current frame.',
        '', 'Private artifacts include immutable per-group decisions, support metadata, results, identity, lock, heartbeat and events.',
        'Re-running evaluation requires `--resume`. The full legacy test suite is not claimed here.', '',
        '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode batch --phase decide',
        '.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode fitting --phase decide',
        '.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode batch --phase evaluate --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode fitting --phase evaluate --resume',
        '.venv-pytorch/bin/python scripts/report_m3w_european_causal_abstention.py', '```', '',
        'Existing authorized source data and private score banks are required; aggregate-only Git contents cannot recreate them.',
        'Independent model selection, risk calibration and confirmation remain closed. Stage5C/SMC remain off.',
        'Image-pixel obs8/pred12 rawstride12; not historical raw-t50, metric/seconds, human gold, true 3D or foundation evidence.']
    (run.PUBLIC/'execution_notes.md').write_text('\n'.join(checks)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, parent_name in zip(axes, ('cv_targets', 'floor_both')):
        for i, name in enumerate(('stop', 'support', 'combined')):
            values = [r['views'][parent_name+'__'+name]['ADE_vs_original']['all']['equal_scene_gain_percent']
                      for groups in rows.values() for r in groups.values()]
            ax.scatter(np.full(len(values), i), values, s=13, alpha=.6, label=name)
        ax.axhline(0, color='black', linewidth=.7); ax.set_xticks(range(3), ('Stop', 'Support', 'Combined'))
        ax.set_title(parent_name); ax.set_ylabel('All-ADE gain vs unchanged parent (%)')
    fig.suptitle('All 36 dependent development views; no selected winner')
    fig.savefig(run.PUBLIC/'guard_changes.svg'); fig.savefig(run.PRIVATE/'guard_changes.png', dpi=160); plt.close(fig)


def main():
    rows, refs, counts, barrier = verify()
    decision_context(rows)
    summary = publish(rows); documents(summary, counts, barrier, rows)
    test_log = run.PRIVATE/'scoped_tests.log'
    files = json.loads((run.PRIVATE/'scoped_test_files.json').read_text())
    match = re.search(r'(\d+) passed', test_log.read_text())
    if not match or ' failed' in test_log.read_text() or len(files) != 50:
        raise ValueError('Complete scoped tests required')
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    phases = []
    for start in events:
        if start['state'] != 'phase_started': continue
        ends = [e for e in events if e['state'] == 'phase_complete' and e.get('mode') == start['mode']
                and e.get('phase') == start['phase'] and e['utc'] >= start['utc']]
        if not ends: raise ValueError('Required phase remains unfinished')
        end = min(ends, key=lambda e: e['utc'])
        phases.append(dict(mode=start['mode'], phase=start['phase'], pid=start['pid'], start=start['utc'], end=end['utc'],
            elapsed_seconds=(datetime.fromisoformat(end['utc'])-datetime.fromisoformat(start['utc'])).total_seconds()))
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    resources = dict(phases=phases, new_neural_training=False, runtime='native_arm64_cpu_four_threads_interop_one_workers_zero',
        create_queue=dict(checked_start_utc=queue['started_utc'], checked_end_utc=queue['completed_utc'],
            successful_readonly_query=queue['response']['returncode'] == 0,
            jobs_submitted=queue['jobs_submitted'], remote_modified=queue['remote_modified'],
            private_receipt_sha256=run.digest(run.PRIVATE/'create_queue.json')),
        remote_artifact_store='not_inspected_no_absence_claim', local_resource_failure=False)
    dump(run.PUBLIC/'compute_receipt.json', resources)
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, checks=counts, group_receipts=refs,
        decision_barrier=barrier, reporter_sha256=run.digest(Path(__file__)),
        summary_sha256=run.digest(run.PUBLIC/'summary_metrics.json'),
        tests_passed=int(match.group(1)), scoped_test_files=files,
        test_source_hashes={f: run.digest(ROOT/f) for f in files}, test_log_sha256=run.digest(test_log),
        full_legacy_suite_run=False,
        artifact_hashes={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*')
                        if p.is_file() and p.name != 'completion_checks.json'},
        new_neural_training=False, deployment_changed=False, independent_confirmation=False))
    print(json.dumps(dict(verified=True, **counts)), flush=True)


if __name__ == '__main__': main()
