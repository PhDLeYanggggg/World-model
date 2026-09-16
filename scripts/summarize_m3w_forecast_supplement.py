"""Publish compact supplementary tables without copying row-level caches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest


def tables(report):
    if report['identity']['eligible_for_selection'] is not False or report['scope'] != 'development_diagnostic_only':
        raise ValueError('Only fixed diagnostic results are allowed')
    if set(report['seeds']) != {'17', '29', '43'}:
        raise ValueError('All three registered seeds are required')
    rows, matched = [], []
    for seed, candidates in report['seeds'].items():
        expected = {f'seed{seed}_{h}_{p}' for h in ('ridge', 'neural_cost') for p in ('conservative', 'moderate')}
        if set(candidates) != expected:
            raise ValueError('Missing or additional unregistered diagnostic candidate')
        for name, result in candidates.items():
            prefix = result['raw50_prefix']
            if prefix['status'] != 'scored_exact_prefix_not_reconditioned_short_horizon':
                rows.append({'seed': int(seed), 'candidate': name, 'status': prefix['status']})
            else:
                for arm, subsets in prefix['arms'].items():
                    for subset, metrics in subsets.items():
                        for metric, value in metrics.items():
                            rows.append({'seed': int(seed), 'candidate': name, 'arm': arm, 'subset': subset,
                                'metric': metric, 'status': value['status'], 'count': value['count'],
                                **{k: value.get(k) for k in ('baseline_error', 'selected_error', 'improvement_pct',
                                    'degradation_fraction', 'mean_excess_over_floor', 'switch_rate_valid_labels')},
                                'bootstrap': value.get('bootstrap')})
            for population, metrics in result['matched_control_comparisons'].items():
                for metric, value in metrics.items():
                    matched.append({'seed': int(seed), 'candidate': name, 'population': population,
                        'metric': metric, **value, 'scene_query_status_counts': result['exact_count_status_counts_scene_queries']})
    groups = {}
    for row in rows:
        if row['status'] != 'evaluated' or row['improvement_pct'] is None:
            continue
        suffix = row['candidate'].removeprefix(f"seed{row['seed']}_")
        key = (suffix, row['arm'], row['subset'], row['metric'])
        groups.setdefault(key, []).append(row['improvement_pct'])
    seeds = [{'candidate_family': k[0], 'arm': k[1], 'subset': k[2], 'metric': k[3],
              'seeds_with_defined_gain': len(v), 'mean_gain_pct': statistics.mean(v),
              'seed_sd_percentage_points': statistics.stdev(v) if len(v) >= 2 else None,
              'scene_ci': None} for k, v in sorted(groups.items())]
    return {'raw50_rows': rows, 'exact_count_rows': matched, 'seed_descriptive_statistics': seeds,
            'seed_sd_is_not_scene_uncertainty': True, 'no_deployment_selected': True}


def number(value):
    return 'not_run' if value is None else f'{value:.6g}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.cache_dir / 'supplement_report.json'
    receipt = json.loads((args.cache_dir / 'completion.json').read_text())
    if file_digest(source) != receipt['report_sha256']:
        raise ValueError('Supplement report changed after completion')
    report = json.loads(source.read_text())
    result = tables(report)
    result.update(result_source='fresh_run_summary_of_verified_completed_supplement',
        source_sha256=file_digest(source), protocol_sha256=report['identity']['parent_protocol_sha256'],
        source_identity=report['identity'], source_runtime_seconds=report['elapsed_seconds_this_invocation'],
        independent_confirmation=False, stage5c_executed=False, smc_enabled=False)
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    fields = ['seed', 'candidate', 'arm', 'subset', 'metric', 'status', 'count', 'baseline_error',
              'selected_error', 'improvement_pct', 'degradation_fraction', 'mean_excess_over_floor', 'switch_rate_valid_labels']
    with (out / 'raw50.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['raw50_rows'])
    lines = ['# Fixed-Forecast Supplement', '',
        'Development-only, all three seeds and every frozen cost-head/policy combination. No additional model selection.',
        'Raw50 is the exact native-grid prefix of the original 12-step prediction. Its scale is not refitted.',
        'Primary full-horizon evaluation remains authoritative; this table cannot rescue a failed primary result.', '',
        '| Seed | Candidate | Arm | Raw50 ADE gain % | Raw50 FDE gain % | Full-path easy degradation % (prefix ADE) |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    rows = result['raw50_rows']
    for row in rows:
        if row.get('subset') != 'all' or row.get('metric') != 'ade':
            continue
        matching = [r for r in rows if all(r.get(k) == row.get(k) for k in ('seed', 'candidate', 'arm'))]
        fde = next(r for r in matching if r.get('subset') == 'all' and r.get('metric') == 'fde')
        easy = next(r for r in matching if r.get('subset') == 'full_path_defined_easy' and r.get('metric') == 'ade')
        d = easy.get('degradation_fraction')
        lines.append(f"| {row['seed']} | {row['candidate']} | {row['arm']} | {number(row.get('improvement_pct'))} | {number(fde.get('improvement_pct'))} | {number(d*100 if d is not None else None)} |")
    lines += ['', '## Matched Nonzero Intervention Counts', '',
        '| Seed | Candidate | ADE joint minus independent | Scored agent queries | Matched nonzero scene queries | Solver/unmatched queries |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result['exact_count_rows']:
        if row['population'] != 'matched_nonzero_only' or row['metric'] != 'ade':
            continue
        counts = row['scene_query_status_counts']
        lines.append(f"| {row['seed']} | {row['candidate']} | {number(row.get('left_minus_right_error'))} | {row['count']} | {counts.get('matched_nonzero', 0)} | {counts.get('not_matched_solver_or_feasibility_failure', 0)} |")
    lines += ['', 'Negative differences favor joint selection on the same frozen candidates. Count matching applies to all past-supported agents before label access; scored-only coverage need not match after incomplete labels are filtered. Zero-count queries are reported in JSON but are not evidence of coordination.',
        'One physical development site does not permit an informative scene-bootstrap CI. Seed SD describes training randomness only. Coordinates and time remain unverified dataset-local/native-frame units. No formal safety or deployment claim.', '']
    (out / 'results.md').write_text('\n'.join(lines))
    print(json.dumps({'raw50_table_rows': len(rows), 'matched_table_rows': len(result['exact_count_rows'])}))


if __name__ == '__main__':
    main()
