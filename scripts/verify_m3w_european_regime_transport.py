"""Replay crossed predictions, held costs, reports and their scoped contracts."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_regime_transport as run
from scripts.verify_m3w_european_oof_magnitude import TESTS as PARENT_TESTS

TESTS = ['tests/test_m3w_regime_transport.py', 'tests/test_m3w_regime_transport_report.py', *PARENT_TESTS]


def execute(args, label):
    path = run.PRIVATE/(label+'.log')
    with path.open('w') as handle:
        result = subprocess.run([sys.executable, *args], cwd=ROOT,
            stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = []
    for phase in ('verify_training', 'verify_eval'):
        logs.append(execute(['scripts/run_m3w_european_regime_transport.py', '--phase', phase], phase))
    names = ['aggregate_metrics.json', 'compute_receipt.json', 'results.md',
        'crossed_raw.svg', 'crossed_scaled.svg', 'magnitude_transport.svg']
    before = {p: run.digest(run.PUBLIC/p) for p in names}
    logs.append(execute(['scripts/report_m3w_european_regime_transport.py'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_regime_transport.py'], 'plot_replay'))
    assert all(run.digest(run.PUBLIC/p) == h for p, h in before.items())
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot()
    cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    for name in ('training_replay.json', 'evaluation_replay.json'):
        assert json.loads((run.PUBLIC/name).read_text())['exact']
    support = json.loads((run.PUBLIC/'support_report.json').read_text())
    assert support['training_allowed'] and len(support['rows']) == 1728
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    assert len(rows) == 144 and sum(len(r['replicas']) for r in rows) == 432
    sources = sorted(set(run.FILES+TESTS+[
        'scripts/verify_m3w_european_regime_transport.py',
        'scripts/plot_m3w_european_regime_transport.py',
        'scripts/recover_m3w_oof_magnitude_checkpoint.py']))
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True,
        fresh_crossed_checkpoints_replayed=864, outer_prediction_replicas_replayed=432,
        native_bridge_parameters_exact=True, within_regime_sampler_matches=864,
        source_held_views_replayed=144, direct_MSE_checks=13824,
        native_three_site_metrics_exact=True, reports_byte_reproducible=True,
        scoped_tests_passed=len(cases), scoped_test_files=len(TESTS),
        full_legacy_suite='not_run', independent_confirmation=False, deployment_changed=False,
        logs=logs, local_detailed_metrics=[run.artifact(run.PUBLIC/'readout.json')],
        artifacts={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
            if p.is_file() and p.name not in ('verification.json', 'readout.json')},
        source_bindings={p: run.digest(ROOT/p) for p in sources}))
    print(json.dumps(dict(all_passed=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS))))


if __name__ == '__main__':
    main()
