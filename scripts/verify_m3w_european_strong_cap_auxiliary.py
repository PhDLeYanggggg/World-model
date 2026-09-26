"""Replay the strong-control reconstruction, all checkpoints and complete readout."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_strong_cap_auxiliary as run
from scripts.verify_m3w_european_cap_auxiliary_cost import TESTS as PARENT_TESTS
TESTS = ['tests/test_m3w_strong_cap_auxiliary.py', *PARENT_TESTS]


def execute(args, name):
    path = run.PRIVATE/(name+'.log')
    with path.open('w') as handle:
        process = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError(f'{name} failed: {path}')
    return run.artifact(path)


def main():
    logs = []
    for phase in ['verify_training', 'verify_eval']:
        logs.append(execute(['scripts/run_m3w_european_strong_cap_auxiliary.py', '--phase', phase], phase))
    names = ['aggregate_metrics.json', 'results.md', 'strong_cap_auxiliary_contrasts.svg',
             'training_metrics.json', 'training_loss_endpoints.csv', 'compute_receipt.json', 'fit_held_diagnosis.json']
    before = {n: run.digest(run.PUBLIC/n) for n in names}
    logs.append(execute(['scripts/report_m3w_european_strong_cap_auxiliary.py'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_strong_cap_auxiliary.py'], 'figure_replay'))
    for n, h in before.items(): assert run.digest(run.PUBLIC/n) == h
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    a = json.loads((run.PUBLIC/'training_replay.json').read_text())
    b = json.loads((run.PUBLIC/'readout_replay.json').read_text())
    assert a['heads'] == 432 and a['updates'] == 864000 and a['matched_arm_checks'] == 288
    assert len(a['controls']) == 144 and all(c['tolerance_pass'] for c in a['controls'])
    assert b == dict(groups=36, views=144, exact=True, direct_MSE_checks=1152)
    sources = set(run.FILES+TESTS+['scripts/verify_m3w_european_strong_cap_auxiliary.py'])
    doc = dict(all_passed=True, heads_replayed=432, readout_views_replayed=144,
        strong_controls_reconstructed=144, exact_strong_controls=sum(c['exact'] for c in a['controls']),
        matched_arm_checks=288, direct_MSE_checks=1152, tests_passed=len(cases), scoped_test_files=len(TESTS),
        full_legacy_suite='not_run', independent_retraining='not_run', report_figure_byte_reproducible=True,
        logs=logs, artifacts={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
                             if p.is_file() and p.name != 'verification.json'},
        source_bindings={p: run.digest(ROOT/p) for p in sorted(sources)})
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ('logs', 'artifacts', 'source_bindings')}, indent=2))


if __name__ == '__main__': main()
