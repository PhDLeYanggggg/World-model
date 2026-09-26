"""Replay every fitting diagnostic, generated report and scoped regression test."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_gradient as run
from scripts.verify_m3w_european_strong_cap_auxiliary import TESTS as PARENT_TESTS
TESTS = ['tests/test_m3w_aux_gradient.py', 'tests/test_m3w_aux_gradient_report.py', *PARENT_TESTS]


def execute(args, label):
    path = run.PRIVATE/(label+'.log')
    with path.open('w') as handle:
        process = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = [execute(['scripts/run_m3w_european_aux_gradient.py', '--phase', 'verify'], 'replay')]
    names = ['aggregate_metrics.json', 'results.md', 'adamw_probe_effects.svg', 'secondary_diagnosis.json']
    before = {n: run.digest(run.PUBLIC/n) for n in names}
    logs.append(execute(['scripts/report_m3w_european_aux_gradient.py'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_aux_gradient.py'], 'figure_replay'))
    logs.append(execute(['scripts/diagnose_m3w_european_aux_gradient.py'], 'diagnosis_replay'))
    assert all(run.digest(run.PUBLIC/n) == h for n, h in before.items())
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    replay = json.loads((run.PUBLIC/'replay.json').read_text())
    freeze = json.loads((run.PUBLIC/'diagnostic_freeze.json').read_text())
    assert replay['exact_replay'] and not freeze['exact_replay']
    assert {k: v for k, v in replay.items() if k != 'exact_replay'} == {k: v for k, v in freeze.items() if k != 'exact_replay'}
    unknown = overlap = updates = 0
    for ref in freeze['receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        row = json.loads((ROOT/ref['path']).read_text())
        for state in row['states']:
            for repeat in state['repeats']:
                updates += len(repeat['after'])
                unknown += repeat['update_unknown_rows']; overlap += repeat['row_overlap_count']
    assert updates == 17280 and unknown == overlap == 0
    source = sorted(set(run.FILES + TESTS + ['scripts/plot_m3w_european_aux_gradient.py',
        'scripts/verify_m3w_european_aux_gradient.py', 'scripts/diagnose_m3w_european_aux_gradient.py']))
    doc = dict(all_passed=True, exact_virtual_update_replay=True, virtual_updates=updates,
        views=144, final_states=432, unknown_update_rows=unknown, update_probe_row_overlap=overlap,
        scoped_tests_passed=len(cases), scoped_test_files=len(TESTS), full_legacy_suite='not_run',
        new_full_training=False, generalization_verified=False, reports_figure_byte_reproducible=True,
        logs=logs, artifacts={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
                             if p.is_file() and p.name != 'verification.json'},
        source_bindings={p: run.digest(ROOT/p) for p in source})
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ('logs', 'artifacts', 'source_bindings')}, indent=2))


if __name__ == '__main__': main()
