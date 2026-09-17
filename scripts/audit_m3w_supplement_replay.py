"""Check supplementary scoring against every completed primary row export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_development_evaluation import content_digest
from scripts.evaluate_m3w_forecast_supplement import verify_completed_evaluation


def compare_rows(primary, supplement):
    if len(primary) != len(supplement):
        raise ValueError('Query population changed')
    result = {'agent_queries': len(primary), 'fixed_forecast_rows_different': 0,
              'ordinary_control_rows_different': 0, 'max_forecast_error_absolute_difference': 0.,
              'count_control_queries_available': 0, 'count_control_switch_disagreements': 0}
    for before, after in zip(primary, supplement):
        if {k: v for k, v in before.items() if k != 'arms'} != {k: v for k, v in after.items() if k != 'arms'}:
            raise ValueError('Query identity, scale or labels changed')
        fixed_different = False
        for arm in ('floor', 'uncontrolled'):
            fixed_different |= before['arms'][arm] != after['arms'][arm]
            for metric in ('ade', 'fde'):
                a, b = before['arms'][arm][metric], after['arms'][arm][metric]
                if (a is None) != (b is None):
                    raise ValueError('Forecast label availability changed')
                if a is not None:
                    result['max_forecast_error_absolute_difference'] = max(
                        result['max_forecast_error_absolute_difference'], abs(a-b))
        result['fixed_forecast_rows_different'] += int(fixed_different)
        result['ordinary_control_rows_different'] += int(any(
            before['arms'][arm] != after['arms'][arm] for arm in ('independent', 'scene_uniform', 'joint')))
        controls = after['arms']
        if all(k in controls for k in ('independent_count_reference', 'joint_exact_count')):
            result['count_control_queries_available'] += 1
            result['count_control_switch_disagreements'] += int(
                controls['independent_count_reference']['switch'] != controls['joint_exact_count']['switch'])
    result['fixed_forecast_scoring_identical'] = result['fixed_forecast_rows_different'] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--supplement-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.supplement_dir
    completion = json.loads((root/'completion.json').read_text())
    identity = json.loads((root/'run_identity.json').read_text())
    if completion != {'run_sha256': content_digest(identity), 'report_sha256': file_digest(root/'supplement_report.json')}:
        raise ValueError('Supplement completion identity mismatch')
    rows, hashes = [], {}
    for plan in identity['plans']:
        for candidate in plan['candidates']:
            seed = int(candidate['id'].split('_')[0].removeprefix('seed'))
            directory = args.study_dir/f'seed{seed}_development'
            verify_completed_evaluation(directory, file_digest)
            primary_identity = json.loads((directory/'run_identity.json').read_text())
            if primary_identity['plan'] != plan or primary_identity['protocol_sha256'] != identity['parent_protocol_sha256']:
                raise ValueError('Primary/supplement plan mismatch')
            for recording in ('ucy_students01', 'ucy_students03'):
                key = [candidate['id'], recording]
                original = directory/(content_digest(key)+'.rows.json')
                original_receipt = json.loads((directory/(content_digest(key)+'.receipt.json')).read_text())
                if original_receipt != {'key': key, 'run_sha256': content_digest(primary_identity), 'cache_sha256': file_digest(original)}:
                    raise ValueError('Primary row cache changed')
                key = [seed, *key]
                replay = root/(content_digest(key)+'.json')
                replay_receipt = json.loads((root/(content_digest(key)+'.receipt.json')).read_text())
                if replay_receipt != {'key': key, 'run_sha256': content_digest(identity), 'cache_sha256': file_digest(replay)}:
                    raise ValueError('Supplement row cache changed')
                result = compare_rows(json.loads(original.read_text()), json.loads(replay.read_text())['full'])
                rows.append({'seed': seed, 'candidate': candidate['id'], 'recording': recording, **result})
                hashes[str(original)] = file_digest(original)
                hashes[str(replay)] = file_digest(replay)
    report = {'result_source': 'fresh_run_comparison_of_hash_verified_primary_and_supplement',
              'rows': rows, 'source_sha256': hashes, 'analysis_code_sha256': file_digest(Path(__file__)),
              'all_fixed_forecast_scoring_identical': all(r['fixed_forecast_scoring_identical'] for r in rows),
              'ordinary_control_rows_different': sum(r['ordinary_control_rows_different'] for r in rows),
              'count_control_queries_available': sum(r['count_control_queries_available'] for r in rows),
              'count_control_switch_disagreements': sum(r['count_control_switch_disagreements'] for r in rows),
              'scope': 'exported_errors_and_decisions_not_full_tensor_bitwise_replay',
              'independent_confirmation': False}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir/'replay_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'source_sha256')}))
    return 0 if report['all_fixed_forecast_scoring_identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
