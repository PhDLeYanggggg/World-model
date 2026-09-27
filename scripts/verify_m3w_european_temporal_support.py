"""Single-use attestation of numerical replay and focused regression checks."""
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_temporal_support as run
from scripts.verify_m3w_european_cost_shape import TESTS as PARENT_TESTS
TESTS = ['tests/test_m3w_temporal_support.py', 'tests/test_m3w_temporal_support_report.py', *PARENT_TESTS]


def execute(args, label, directory):
    path = directory/(label+'.log')
    with path.open('x') as handle:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    if (run.PUBLIC/'verification.json').exists():
        raise ValueError('Immutable attestation exists; inspect hashes or use runner replay phases directly')
    directory = run.PRIVATE/'verification_runs'/str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=False); logs = []
    for phase in ('verify_fit', 'verify_eval'):
        logs.append(execute(['scripts/run_m3w_european_temporal_support.py', '--phase', phase], phase, directory))
    names = ['aggregate_metrics.json', 'gates.json', 'results.md', 'context_intervals.svg', 'event_support.svg',
        'support_diagnostics.json', 'support_diagnostics.md']
    before = {p:run.digest(run.PUBLIC/p) for p in names}
    logs.append(execute(['scripts/report_m3w_european_temporal_support.py'], 'report_replay', directory))
    logs.append(execute(['scripts/plot_m3w_european_temporal_support.py'], 'plot_replay', directory))
    logs.append(execute(['scripts/diagnose_m3w_european_temporal_support.py'], 'diagnosis_replay', directory))
    assert all(run.digest(run.PUBLIC/p) == value for p, value in before.items())
    xml = directory/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests', directory))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+key) for key in ('failure', 'error', 'skipped'))
    for name in ('fitting_replay.json', 'evaluation_replay.json'):
        assert json.loads((run.PUBLIC/name).read_text())['exact']
    sources = sorted(set(run.FILES+TESTS+['scripts/verify_m3w_european_temporal_support.py',
        'scripts/plot_m3w_european_temporal_support.py', 'scripts/diagnose_m3w_european_temporal_support.py']))
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True, ridge_fits_replayed=1728,
        fitting_locality_views_replayed=432, new_neural_heads=0, reports_byte_reproducible=True,
        scoped_tests_passed=len(cases), scoped_test_files=len(TESTS), full_legacy_suite='not_run',
        cold_rebuild_from_new_raw_downloads='not_run', independent_confirmation=False,
        logs=logs, local_detailed_metrics=[run.artifact(run.PRIVATE/'readout.json')],
        artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
            if p.is_file() and p.name != 'verification.json'},
        source_bindings={p:run.digest(ROOT/p) for p in sources}))
    print(json.dumps(dict(all_passed=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS))))


if __name__ == '__main__': main()
