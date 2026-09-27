"""Replay 2160 fitting measurements, final-state checks and generated artifacts."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_trajectory as run
from scripts.verify_m3w_european_aux_gradient import TESTS as PREVIOUS_TESTS
TESTS = ['tests/test_m3w_aux_trajectory.py', 'tests/test_m3w_aux_trajectory_diagnosis.py',
         'tests/test_m3w_aux_initial_prior.py', *PREVIOUS_TESTS]


def execute(args, label):
    path = run.PRIVATE/(label+'.log')
    with path.open('w') as handle:
        process = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = [execute(['scripts/run_m3w_european_aux_trajectory.py', '--phase', 'verify'], 'replay')]
    names = ['aggregate_metrics.json', 'results.md', 'training_time_effects.svg',
             'secondary_diagnosis.json', 'gradient_time_effects.svg', 'initial_prior_posthoc.json']
    before = {n: run.digest(run.PUBLIC/n) for n in names}
    logs.append(execute(['scripts/report_m3w_european_aux_trajectory.py'], 'report_replay'))
    logs.append(execute(['scripts/diagnose_m3w_european_aux_trajectory.py'], 'diagnosis_replay'))
    logs.append(execute(['scripts/inspect_m3w_aux_initial_prior.py'], 'prior_metadata_replay'))
    assert all(run.digest(run.PUBLIC/n) == h for n, h in before.items())
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    replay = json.loads((run.PUBLIC/'replay.json').read_text())
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    assert replay['exact_diagnostic_replay'] and not freeze['exact_diagnostic_replay']
    assert {k:v for k,v in replay.items() if k != 'exact_diagnostic_replay'} == {
        k:v for k,v in freeze.items() if k != 'exact_diagnostic_replay'}
    assert freeze['heads'] == 432 and freeze['snapshots'] == 2160 and freeze['updates'] == 864000
    source = sorted(set(run.FILES+TESTS+['scripts/diagnose_m3w_european_aux_trajectory.py',
        'scripts/verify_m3w_european_aux_trajectory.py', 'scripts/inspect_m3w_aux_initial_prior.py']))
    doc = dict(all_passed=True, reconstructed_heads=432, optimizer_updates=864000, snapshots_replayed=2160,
        exact_original_final_states=True, reports_figures_byte_reproducible=True,
        scoped_tests_passed=len(cases), scoped_test_files=len(TESTS), full_legacy_suite='not_run',
        new_model_variant=False, generalization_verified=False, logs=logs,
        artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
                   if p.is_file() and p.name != 'verification.json'},
        source_bindings={p:run.digest(ROOT/p) for p in source})
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps({k:v for k,v in doc.items() if k not in ('logs', 'artifacts', 'source_bindings')}, indent=2))


if __name__ == '__main__': main()
