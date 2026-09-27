"""Replay repaired heads, source-held costs, aggregate reports and scoped tests."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_prior as run
from scripts.verify_m3w_european_aux_trajectory import TESTS as PREVIOUS_TESTS
TESTS = ['tests/test_m3w_aux_prior.py', 'tests/test_m3w_aux_prior_report.py', *PREVIOUS_TESTS]


def execute(args, label):
    path = run.PRIVATE/(label+'.log')
    with path.open('w') as handle:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = [execute(['scripts/run_m3w_european_aux_prior.py', '--phase', 'verify_training'], 'training_replay'),
            execute(['scripts/run_m3w_european_aux_prior.py', '--phase', 'verify_eval'], 'readout_replay')]
    names = ['aggregate_metrics.json', 'results.md', 'training_endpoints.json', 'compute_receipt.json', 'source_held_costs.svg']
    before = {p:run.digest(run.PUBLIC/p) for p in names}
    logs.append(execute(['scripts/report_m3w_european_aux_prior.py'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_aux_prior.py'], 'plot_replay'))
    assert all(run.digest(run.PUBLIC/p) == h for p,h in before.items())
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    train = json.loads((run.PUBLIC/'training_replay.json').read_text())
    readout = json.loads((run.PUBLIC/'readout_replay.json').read_text())
    assert train['exact'] and train['heads'] == 288 and readout['exact'] and readout['views'] == 144
    sources = sorted(set(run.FILES+TESTS+['scripts/verify_m3w_european_aux_prior.py', 'scripts/plot_m3w_european_aux_prior.py']))
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True,
        heads_replayed=288, initialization_checks=288, source_held_views_replayed=144,
        reports_byte_reproducible=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS),
        full_legacy_suite='not_run', independent_confirmation=False, deployment_changed=False,
        logs=logs, local_detailed_metrics=[run.artifact(run.PUBLIC/'readout.json')],
        artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
            if p.is_file() and p.name not in ('verification.json', 'readout.json')},
        source_bindings={p:run.digest(ROOT/p) for p in sources}))
    print(json.dumps(dict(all_passed=True, heads_replayed=288, source_held_views_replayed=144,
        scoped_tests_passed=len(cases), scoped_test_files=len(TESTS), full_legacy_suite='not_run')))


if __name__ == '__main__': main()
