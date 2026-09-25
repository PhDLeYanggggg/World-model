"""Check and package the completed source experiment without selecting a model."""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a = json.loads((PUBLIC/'analysis.json').read_text())
    for name in ('verification.json', 'checkpoint_replay.json', 'accounting_audit.json'):
        r = json.loads((PUBLIC/name).read_text())
        if not r['all_passed'] or r['analysis_sha256'] != sha(PUBLIC/'analysis.json'):
            raise ValueError('Verified complete readout required')
    producer = json.loads((PUBLIC/'producer_summary.json').read_text())
    if not producer['exact_replay'] or producer['new_updates'] != 72000:
        raise ValueError('Complete producer replay required')
    pids = [36794, 36827, 39574, 39666, 39696, 40172]
    for pid in pids:
        p = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
        if 'run_m3w_european_nested_' in p.stdout:
            raise ValueError('Required experiment process is still active')
    old = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_risk_v1/completion_checks.json'
    tests = json.loads(old.read_text())['scoped_test_files']+[
        'tests/test_m3w_nested_calibration.py', 'tests/test_m3w_source_risk_calibration.py',
        'tests/test_m3w_nested_calibration_reporting.py']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT,
        capture_output=True, text=True)
    (PUBLIC/'scoped_tests.txt').write_text(result.stdout+result.stderr)
    if result.returncode or not re.search(r'\b196 passed\b', result.stdout):
        raise ValueError('Scoped tests failed; inspect scoped_tests.txt')
    groups = {}
    for candidate in ('neural', 'damping097'):
        groups[candidate] = {}
        for rule in ('none', 'population_rescale', 'selected_risk_grid'):
            rows = [p for key, p in a['policies'].items() if f'_{candidate}_' in key and key.endswith('_'+rule)]
            assert len(rows) == 12
            groups[candidate][rule] = dict(views=12,
                full_observed_safety=sum(p['safety_observed_pass'] for p in rows),
                gain_range_percent=[min(p['ADE_vs_CV']['equal_scene_gain_percent'] for p in rows),
                                    max(p['ADE_vs_CV']['equal_scene_gain_percent'] for p in rows)])
    contrasts = {}
    for subset in ('all', 'hard', 'easy'):
        rows = [p[subset] for p in a['neural_vs_damping'].values()]
        contrasts[subset] = dict(comparisons=len(rows),
            positive_points=sum(p['equal_scene_gain_percent'] > 0 for p in rows),
            positive_CI=sum(p['scene_bootstrap_ci95'][0] > 0 for p in rows),
            negative_CI=sum(p['scene_bootstrap_ci95'][1] < 0 for p in rows))
    output = dict(result_source='fresh_run_completion_checks_cached_verified_models',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        analysis_sha256=sha(PUBLIC/'analysis.json'), summary_sha256=sha(PUBLIC/'summary_metrics.json'),
        code_sha256=sha(Path(__file__)), python=sys.version, architecture=platform.machine(),
        scoped_test_files=tests, scoped_tests_passed=196, full_legacy_suite_run=False,
        scoped_test_log_sha256=sha(PUBLIC/'scoped_tests.txt'),
        new_producer_updates=72000, new_head_updates=108000, new_total_updates=180000,
        producers_replayed=18, heads_replayed=54, matched_sampler_groups=9,
        heads_per_sampler_group=6, decisions_independently_reconstructed=216,
        calibration_maps=72, pointwise_views=72, groups=groups, neural_vs_damping=contrasts,
        required_processes_finished=pids, all_required_local_processes_finished=True,
        remote_execution_this_experiment='not_run_local_resources_sufficient',
        remote_live_status='not_run_no_new_authenticated_observation',
        reserved_roles_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    (PUBLIC/'completion_checks.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(tests_passed=196, producers_replayed=18, heads_replayed=54,
        new_total_updates=180000, processes_finished=True, neural_advantage=contrasts)))


if __name__ == '__main__':
    main()
