"""Check fixed-model transport evidence without choosing a producer or threshold."""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_producer_transport_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrast_counts(rows):
    points = [r['equal_scene_gain_percent'] for r in rows]
    intervals = [r['scene_bootstrap_ci95'] for r in rows]
    defined = [p for p in points if p is not None]
    return dict(comparisons=len(rows), undefined_points=sum(p is None for p in points),
        positive_points=sum(p > 0 for p in defined),
        positive_CI=sum(c is not None and c[0] > 0 for c in intervals),
        negative_CI=sum(c is not None and c[1] < 0 for c in intervals),
        gain_range_percent=[min(defined), max(defined)] if defined else None)


def main():
    a = json.loads((PUBLIC/'analysis.json').read_text())
    v = json.loads((PUBLIC/'verification.json').read_text())
    b = json.loads((PUBLIC/'bank_replay.json').read_text())
    s = json.loads((PUBLIC/'summary_metrics.json').read_text())
    raw_metrics = json.loads((PUBLIC/'raw_forecast_metrics.json').read_text())
    if not v['all_passed'] or v['analysis_sha256'] != sha(PUBLIC/'analysis.json'):
        raise ValueError('Complete metric recomputation required')
    if s['analysis_sha256'] != v['analysis_sha256'] or b['identity'] != a['identity']:
        raise ValueError('Mismatched reporting or replay lineage')
    if raw_metrics['analysis_sha256'] != v['analysis_sha256'] or len(raw_metrics['metrics']) != 36:
        raise ValueError('All raw ADE/FDE views required')
    if len(b['checks']) != 18 or any(not r['exact'] or r['replay_rows'] != 4096 for r in b['checks']):
        raise ValueError('Every producer requires an exact sampled inference replay')
    if sum(r['rows'] for r in b['checks']) != 3827628 or len(a['views']) != 72 or len(a['small_vs_full']) != 36:
        raise ValueError('Incomplete registered matrix')
    if a['original_decisions_reconstructed'] != 72:
        raise ValueError('Original fixed policies must reproduce')
    for path, expected in a['identity']['bindings'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen transport binding changed: '+path)
    previous = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'
    if sha(previous/'analysis.json') != a['identity']['previous_analysis_sha256']:
        raise ValueError('Starting evidence changed')
    for r in a['lineages']:
        if set(r['trained']) & set(r['readout']):
            raise ValueError('Producer/readout exposure')
        for name in ('checkpoint', 'prediction'):
            if sha(ROOT/r[name]['path']) != r[name]['sha256']:
                raise ValueError('Inference artifact changed')
    events = [json.loads(line) for line in (PRIVATE/'events.jsonl').read_text().splitlines()]
    last_by_pid = {e['pid']: e for e in events}
    for pid, event in last_by_pid.items():
        command = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
        if 'run_m3w_european_producer_transport.py' in command.stdout or event['state'] != 'phase_complete':
            raise ValueError('A required phase remains active or incomplete')
    tests = json.loads((previous/'completion_checks.json').read_text())['scoped_test_files']+[
        'tests/test_m3w_producer_transport.py', 'tests/test_m3w_producer_transport_reporting.py']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT,
        capture_output=True, text=True)
    (PUBLIC/'scoped_tests.txt').write_text(result.stdout+result.stderr)
    if result.returncode or not re.search(r'\b201 passed\b', result.stdout):
        raise ValueError('Scoped tests failed; inspect scoped_tests.txt')
    contrasts = {subset: contrast_counts([r['policy_small_vs_full'][subset]
        for r in a['small_vs_full'].values()]) for subset in ('all', 'hard', 'easy')}
    raw = [r['raw_small_vs_full'] for key, r in a['small_vs_full'].items() if key.endswith('_all')]
    contrasts['raw_distinct_pairs'] = contrast_counts(raw)
    groups = {}
    for candidate in ('full4', 'half0', 'half1', 'damping097'):
        groups[candidate] = {}
        for event in ('all', 'easy'):
            rows = [r for key, r in a['views'].items() if key.endswith('_'+candidate+'_'+event)]
            if len(rows) != 9:
                raise ValueError('All predefined fold/seed views required')
            gains = [r['ADE_vs_CV']['all']['equal_scene_gain_percent'] for r in rows]
            groups[candidate][event] = dict(views=9,
                observed_safety_pass=sum(r['safety_observed_pass'] for r in rows),
                zero_switch_views=sum(r['switch_rate'] == 0 for r in rows),
                gain_range_percent=[min(gains), max(gains)])
    seconds = sum(json.loads(p.read_text())['prediction_seconds']
        for p in (PRIVATE/'predictions').glob('*.json'))
    output = dict(result_source='fresh_run_completion_cached_verified_frozen_models',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        analysis_sha256=v['analysis_sha256'], summary_sha256=sha(PUBLIC/'summary_metrics.json'),
        raw_forecast_sha256=sha(PUBLIC/'raw_forecast_metrics.json'), raw_ADE_FDE_views=36,
        code_sha256=sha(Path(__file__)), python=sys.version, architecture=platform.machine(),
        new_training=False, new_optimizer_updates=0, threshold_refit=False,
        new_prediction_banks=18, prediction_rows=3827628, bank_inference_seconds=seconds,
        producer_replays=18, sampled_rows_per_replay=4096,
        original_decisions_independently_reconstructed=72, pointwise_views=72,
        paired_comparisons=36, distinct_raw_comparisons=18, groups=groups, contrasts=contrasts,
        scoped_test_files=tests, scoped_tests_passed=201, full_legacy_suite_run=False,
        scoped_test_log_sha256=sha(PUBLIC/'scoped_tests.txt'),
        required_processes_finished=sorted(last_by_pid), all_required_local_processes_finished=True,
        remote_execution='not_run_local_inference_sufficient',
        remote_queue='cached_verified_fresh_authenticated_readonly_snapshot_20260925_055009UTC',
        remote_m3w_asset_directory='not_run_unverified',
        reserved_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    (PUBLIC/'completion_checks.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(tests_passed=201, banks_replayed=18, comparisons=contrasts, processes_finished=True)))


if __name__ == '__main__':
    main()
