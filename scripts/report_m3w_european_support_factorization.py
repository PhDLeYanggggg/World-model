"""Complete fixed-support factorization readout; no post-readout policy selection."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_support_factorization as run
from scripts.report_m3w_european_causal_abstention import verify as verify_prior, fmt
from scripts.report_m3w_european_floor_relative import spread, dump
from scripts.report_m3w_european_protected_motion import compact_metric
from src.world_model.m3w_support_factorization import GUARDS, REASONS

POLICIES = ('stop',) + tuple(g+s for g in GUARDS for s in ('', '_risk', '_random'))
SUBSETS = ('all', 'easy', 'hard', 'complete')
EXPECTED_COUNTS = dict(saved_decisions_verified=936, old_decision_arrays_exact=288,
    independent_coordinate_arrays=144, independent_metric_reductions=12528,
    old_metrics_exact=1728, partition_equalities=4608)


def verify():
    run.ensure_both_frozen(); verify_prior()
    rows, refs = {}, {}; counts = dict.fromkeys(EXPECTED_COUNTS, 0)
    expected = {p+'__'+g for p in ('cv_targets', 'floor_both') for g in POLICIES}
    for mode in ('batch', 'fitting'):
        identity = json.loads((run.PRIVATE/mode/'identity.json').read_text())
        done = json.loads((run.PRIVATE/mode/'evaluation_complete.json').read_text())
        assert done['identity'] == identity and done['all_passed'] and len(done['groups']) == 18
        rows[mode] = {}; refs[mode] = done['groups']
        for ref in done['groups']:
            if run.artifact(ROOT/ref['path']) != ref: raise ValueError('Changed result')
            r = json.loads((ROOT/ref['path']).read_text())
            assert r['verified'] and r['identity'] == identity and set(r['views']) == expected
            assert r['independent_metric_reductions'] == 348
            for k in counts: counts[k] += r[k]
            for v in r['views'].values():
                assert len(v['ADE_vs_floor']['all']['expected_scenes']) == 8
            rows[mode][r['group']] = r
        assert len(rows[mode]) == 18
    assert counts == EXPECTED_COUNTS
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    first = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    frozen = {m: min(e['utc'] for e in events if e['state'] == 'phase_complete'
                    and e.get('phase') == 'decide' and e['mode'] == m) for m in rows}
    assert all(t < first for t in frozen.values())
    return rows, refs, counts, dict(decisions_completed=frozen, first_new_readout=first)


def summarize(views):
    out = {fam: {s: spread([v[fam][s] for v in views]) for s in SUBSETS}
           for fam in ('ADE_vs_floor', 'ADE_vs_stop')}
    out.update(FDE_vs_floor=spread([v['FDE_vs_floor'] for v in views]),
        easy_vs_CV=spread([v['easy_vs_CV'] for v in views]),
        switch_rate_range=[min(v['switch_rate'] for v in views), max(v['switch_rate'] for v in views)],
        unknown_ADE_switch_range=[min(v['unknown_ADE_switches'] for v in views), max(v['unknown_ADE_switches'] for v in views)],
        zero_CV_harm_views=sum(v['zero_CV']['harmed_rows'] > 0 for v in views),
        zero_CV_supported_views=sum(v['zero_CV']['rows'] > 0 for v in views))
    if 'matched_controls' in views[0]:
        out['matched_controls'] = {c: {s: spread([v['matched_controls'][c][s] for v in views])
            for s in SUBSETS} for c in ('risk', 'random')}
    if 'ADE_vs_joint' in views[0]:
        out['ADE_vs_joint'] = {s: spread([v['ADE_vs_joint'][s] for v in views]) for s in SUBSETS}
    return out


def partition_summary(partitions):
    """Account per locality before ranging across dependent views, never pool rows."""
    result = {}
    for subset in SUBSETS:
        result[subset] = {}
        for name in REASONS:
            records = []
            for p in partitions:
                sites = list(p[subset].values())
                supported = [r for r in sites if r['floor_error_sum'] > 0]
                cats = [r['categories'][name] for r in supported]
                harm = float(np.mean([c['harm_pp'] for c in cats])) if cats else None
                benefit = float(np.mean([c['benefit_pp'] for c in cats])) if cats else None
                records.append(dict(avoided_harm_pp=harm, lost_benefit_pp=benefit,
                    removal_change_pp=harm-benefit if harm is not None else None,
                    indexed_rows=sum(r['categories'][name]['indexed_rows'] for r in sites),
                    known_rows=sum(r['categories'][name]['known_rows'] for r in sites)))
            result[subset][name] = {k: ([min(r[k] for r in records if r[k] is not None),
                                          max(r[k] for r in records if r[k] is not None)]
                                      if any(r[k] is not None for r in records) else None)
                                     for k in records[0]}
            result[subset][name]['actual_removal_category'] = name in REASONS[1:5]
            result[subset][name]['benefit_exceeds_harm_views'] = sum(
                r['removal_change_pp'] is not None and r['removal_change_pp'] < 0 for r in records)
    return result


def decision_context(rows):
    out = {}
    for mode, groups in rows.items():
        out[mode] = {}
        for group in groups:
            with np.load(run.PRIVATE/mode/'decisions'/(group+'.npz'), allow_pickle=False) as z:
                q = z['query']; nq = int(q.max())+1; record = {}
                for parent in ('cv_targets', 'floor_both'):
                    stop = z[parent+'__stop']; count0 = np.bincount(q[stop], minlength=nq)
                    for guard in GUARDS:
                        key = parent+'__'+guard; chosen = z[key]
                        count = np.bincount(q[chosen], minlength=nq)
                        r = dict(indexed_queries=nq, stop_active_queries=int((count0 > 0).sum()),
                            free_choice_queries=int(((count > 0) & (count < count0)).sum()),
                            fully_rejected_queries=int(((count == 0) & (count0 > 0)).sum()))
                        for c in ('risk', 'random'):
                            other = z[key+'_'+c]
                            np.testing.assert_array_equal(count, np.bincount(q[other], minlength=nq))
                            r[c+'_different_rows'] = int((chosen != other).sum())
                        record[key] = r
                out[mode][group] = record
    dump(run.PUBLIC/'decision_context.json', dict(modes=out, result_source='fresh_label_free_decision_accounting'))
    return out


def publish(rows):
    summary = {}
    for mode, groups in rows.items():
        summary[mode] = {event: {k: summarize([r['views'][k] for n, r in groups.items() if n.endswith('_'+event)])
            for k in next(iter(groups.values()))['views']} for event in ('all', 'easy')}
        for group, r in groups.items():
            compact = {}
            for k, v in r['views'].items():
                c = {f: {s: compact_metric(m) for s, m in v[f].items()}
                     for f in ('ADE_vs_floor', 'ADE_vs_stop')}
                c.update(FDE_vs_floor=compact_metric(v['FDE_vs_floor']), easy_vs_CV=compact_metric(v['easy_vs_CV']),
                    **{f: v[f] for f in ('switched_rows', 'switch_rate', 'unknown_ADE_switches', 'zero_CV')})
                if 'matched_controls' in v:
                    c['matched_controls'] = {control: {s: compact_metric(m) for s, m in ms.items()}
                                             for control, ms in v['matched_controls'].items()}
                if 'ADE_vs_joint' in v: c['ADE_vs_joint'] = {s: compact_metric(m) for s, m in v['ADE_vs_joint'].items()}
                compact[k] = c
            dump(run.PUBLIC/mode/(group+'.json'), dict(views=compact, partitions=r['partitions']))
            with (run.PUBLIC/mode/(group+'_localities.csv')).open('w', newline='') as f:
                w = csv.writer(f, lineterminator='\n')
                fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99', 'reference_p95')
                w.writerow(['policy', 'comparison', 'subset', 'locality', *fields])
                for key, v in r['views'].items():
                    ms = [(fam, s, m) for fam in ('ADE_vs_floor', 'ADE_vs_stop', 'ADE_vs_joint')
                          for s, m in v.get(fam, {}).items()]
                    ms += [('FDE_vs_floor', 'endpoint', v['FDE_vs_floor']), ('ADE_vs_CV', 'easy', v['easy_vs_CV'])]
                    ms += [('matched_'+c, s, m) for c, ss in v.get('matched_controls', {}).items() for s, m in ss.items()]
                    for fam, s, m in ms:
                        for site, rr in m['by_scene'].items(): w.writerow([key, fam, s, site, *[rr.get(k) for k in fields]])
    records = [r for groups in rows.values() for r in groups.values()]
    result = dict(result_source='fresh_run_fixed_box_factorization', forecasts_heads_population='cached_verified',
        new_neural_training=False, views=936, conditional_development_only=True, bootstrap_resamples=3000,
        bootstrap_unit='source_locality', bootstrap_seed=39271, modes=summary,
        across_dependent_views={k: summarize([r['views'][k] for r in records]) for k in records[0]['views']},
        rejection_partitions={p: partition_summary([r['partitions'][p] for r in records]) for p in ('cv_targets', 'floor_both')},
        partition_semantics='Only four failure categories are removals; retained/outside-stop costs are hypothetical, not applied actions.',
        deployment_changed=False, independent_confirmation=False)
    dump(run.PUBLIC/'summary_metrics.json', result)
    return result


def documents(summary, counts, barrier, rows):
    pooled = summary['across_dependent_views']
    lines = ['# Fixed-Support Factorization Results', '',
        'All 936 registered development views completed. No thresholds or neural weights refitted.',
        'Forecasts, heads and population: cached_verified. Decisions, factor attribution and evaluations: fresh_run.',
        'Ranges span 36 correlated views per policy; CI counts are not independent tests or corrected significance.', '',
        '| Parent / policy | All ADE gain vs floor (%) | All gain vs stop (%) | Positive / negative CI vs stop | Hard gain vs stop (%) | Worst easy locality gain vs CV (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for key, v in pooled.items():
        m = v['ADE_vs_stop']['all']
        lines.append(f"| {key} | {fmt(v['ADE_vs_floor']['all']['range'])} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {fmt(v['ADE_vs_stop']['hard']['range'])} | {v['easy_vs_CV']['worst_locality']:.4f} |")
    lines += ['', '## Comparison With The Joint Box', '',
        'Recovering benefit relative to a harmful filter is not improvement over the unchanged stop policy.', '',
        '| Policy | All gain vs joint (%) | Positive / negative CI | Complete-label gain vs joint (%) |', '|---|---:|---:|---:|']
    for k, v in pooled.items():
        if 'ADE_vs_joint' in v:
            m = v['ADE_vs_joint']['all']
            lines.append(f"| {k} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {fmt(v['ADE_vs_joint']['complete']['range'])} |")
    lines += ['', '## Same-Recording, Same-Frame Count-Matched Controls', '',
        '| Policy | Control | All gain (%) | Positive / negative CI | Easy gain (%) | Hard gain (%) |', '|---|---|---:|---:|---:|---:|']
    for k, v in pooled.items():
        for c, ms in v.get('matched_controls', {}).items():
            m = ms['all']; lines.append(f"| {k} | {c} | {fmt(m['range'])} | {m['positive_CI']} / {m['negative_CI']} | {fmt(ms['easy']['range'])} | {fmt(ms['hard']['range'])} |")
    lines += ['', '## Rejected Benefit Attribution', '',
        'Mutually exclusive categories 1-4 exactly sum to joint-box removals. Each locality uses the same floor-error denominator.',
        'Avoided harm minus lost benefit is a percentage-point change in gain vs floor, not relative gain vs stop.', '',
        '| Parent | Reason | Known rows / view | Avoided harm (pp) | Lost benefit (pp) | Net removal change (pp) |', '|---|---|---:|---:|---:|---:|']
    for p, part in summary['rejection_partitions'].items():
        for name in REASONS[1:5]:
            r = part['all'][name]
            lines.append(f"| {p} | {name} | {fmt(r['known_rows'])} | {fmt(r['avoided_harm_pp'])} | {fmt(r['lost_benefit_pp'])} | {fmt(r['removal_change_pp'])} |")
    lines += ['', 'Every mode, fold, seed, event, source locality and error tail is retained in adjacent JSON/CSV.',
        'Unknown future labels are not zero error. Complete-window and endpoint metrics retain only their actual label support.',
        'The indexed-agent query cohort is not every visible agent. Forecast-disagreement projections do not prove producer transport causality.',
        'No new deployment or independent confirmation. Image-pixel obs8/pred12 rawstride12, not historical raw-t50 or seconds/metric evidence.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    checks = ['# Execution and Reproducibility', '',
        'Registration commit `8f2aca435820cbc79cae21cdc2da809c73bd06ea` was pushed before decision construction.',
        'Native arm64 CPU4/inter-op1/workers0. No new neural training or support-threshold refit.', '',
        '```json', json.dumps(dict(checks=counts, barrier=barrier), indent=2), '```', '',
        'Both full decision banks froze before the first new outcome readout. Scalar source-set and query-ranking replay',
        'checks every saved policy. Old stop/joint/risk/random decisions and metrics replay exactly.',
        'Source/producer exclusions and train-only support boxes retain the prior audited boundaries.',
        'Private decisions and result receipts are immutable and bound to config/code/schema hashes.', '',
        '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode batch --phase decide',
        '.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode fitting --phase decide',
        '.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode batch --phase evaluate --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode fitting --phase evaluate --resume',
        '.venv-pytorch/bin/python scripts/report_m3w_european_support_factorization.py', '```', '',
        'Authorized private source data and score banks are required. Aggregate-only Git cannot recreate raw predictions.',
        'The scoped test manifest is explicit; no claim that the unrelated legacy suite ran.',
        'Opened sources remain development sources. Independent model selection, calibration and confirmation remain closed.',
        'No human-gold, physical-safety, true-3D or foundation claim. Stage5C/SMC remain off.']
    (run.PUBLIC/'execution_notes.md').write_text('\n'.join(checks)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, parent in zip(axes, ('cv_targets', 'floor_both')):
        for i, g in enumerate(GUARDS):
            values = [r['views'][parent+'__'+g]['ADE_vs_stop']['all']['equal_scene_gain_percent']
                      for gs in rows.values() for r in gs.values()]
            ax.scatter(np.full(len(values), i), values, s=12, alpha=.6)
        ax.axhline(0, color='black', linewidth=.7); ax.set_xticks(range(4), ('History', 'Disagreement', 'Separate', 'Joint'))
        ax.set_title(parent); ax.set_ylabel('All-ADE gain vs unchanged stop policy (%)')
    fig.suptitle('Fixed support projections: all 36 dependent development views')
    fig.savefig(run.PUBLIC/'factor_changes.svg'); fig.savefig(run.PRIVATE/'factor_changes.png', dpi=160); plt.close(fig)
    svg = run.PUBLIC/'factor_changes.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    rows, refs, counts, barrier = verify()
    decision_context(rows); summary = publish(rows); documents(summary, counts, barrier, rows)
    test_log = run.PRIVATE/'scoped_tests.log'; files = json.loads((run.PRIVATE/'scoped_test_files.json').read_text())
    match = re.search(r'(\d+) passed', test_log.read_text())
    if not match or ' failed' in test_log.read_text() or len(files) != 53: raise ValueError('Complete scoped tests required')
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]; phases = []
    for start in events:
        if start['state'] != 'phase_started': continue
        ends = [e for e in events if e['state'] == 'phase_complete' and e.get('mode') == start['mode']
                and e.get('phase') == start['phase'] and e['utc'] >= start['utc']]
        if not ends: raise ValueError('Unfinished required phase')
        end = min(ends, key=lambda e: e['utc'])
        phases.append(dict(mode=start['mode'], phase=start['phase'], pid=start['pid'], start=start['utc'], end=end['utc'],
            elapsed_seconds=(datetime.fromisoformat(end['utc'])-datetime.fromisoformat(start['utc'])).total_seconds()))
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    dump(run.PUBLIC/'compute_receipt.json', dict(phases=phases, new_neural_training=False,
        runtime='native_arm64_cpu_four_threads_interop_one_workers_zero',
        create_queue=dict(checked_start_utc=queue['started_utc'], checked_end_utc=queue['completed_utc'],
            successful_readonly_query=queue['response']['returncode'] == 0,
            jobs_submitted=queue['jobs_submitted'], remote_modified=queue['remote_modified'],
            private_receipt_sha256=run.digest(run.PRIVATE/'create_queue.json')),
        handoff='Existing simulation model restrictions unchanged; authorized M3W artifact path unknown.',
        handoff_private_sha256=run.digest(run.PRIVATE/'create_handoff_update.json'),
        remote_artifact_store='not_inspected_no_absence_claim', local_resource_failure=False))
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, checks=counts, group_receipts=refs,
        decision_barrier=barrier, reporter_sha256=run.digest(Path(__file__)), summary_sha256=run.digest(run.PUBLIC/'summary_metrics.json'),
        tests_passed=int(match.group(1)), scoped_test_files=files, test_source_hashes={f: run.digest(ROOT/f) for f in files},
        test_log_sha256=run.digest(test_log), full_legacy_suite_run=False,
        artifact_hashes={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*')
                        if p.is_file() and p.name != 'completion_checks.json'},
        new_neural_training=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(status='complete', **counts, tests=int(match.group(1)))), flush=True)


if __name__ == '__main__': main()
