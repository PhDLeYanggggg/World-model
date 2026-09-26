"""Replay every frozen head and held readout, then seal exact scoped evidence."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_auxiliary_cost as run

TESTS = ['tests/test_m3w_cap_auxiliary_cost.py', 'tests/test_m3w_cap_auxiliary_cost_reporting.py',
    'tests/test_m3w_cap_auxiliary_cost_diagnosis.py',
    'tests/test_m3w_cap_exceedance.py', 'tests/test_m3w_cap_exceedance_reporting.py',
    'tests/test_m3w_membership_auxiliary.py', 'tests/test_m3w_membership_auxiliary_reporting.py',
    'tests/test_m3w_nested_residual.py', 'tests/test_m3w_risk_conditioned_residual.py']


def execute(args, name):
    path = run.PRIVATE/(name+'.log')
    with path.open('w') as handle:
        process = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'{name} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = []
    source = run.parent.risk.PUBLIC/'support_report.json'
    references = json.loads(source.read_text())['receipts']
    assert len(references) == 144
    checked = 0
    for ref in references:
        assert run.artifact(ROOT/ref['path']) == ref
        receipt = json.loads((ROOT/ref['path']).read_text())
        for artifact in receipt['artifacts'].values():
            assert run.artifact(ROOT/artifact['path']) == artifact
            checked += 1
    run.immutable_json(run.PUBLIC/'reference_source_audit.json', dict(all_passed=True,
        receipt_count=len(references), private_artifacts_checked=checked, source_report=run.artifact(source)))
    for phase in ['verify_training', 'verify_eval']:
        logs.append(execute(['scripts/run_m3w_european_cap_auxiliary_cost.py', '--phase', phase], phase))
    names = ['aggregate_metrics.json', 'results.md', 'cap_auxiliary_cost_contrasts.svg',
             'training_metrics.json', 'training_loss_endpoints.csv', 'compute_receipt.json',
             'fit_held_diagnosis.json', 'fit_held_diagnosis.md']
    before = {name: run.digest(run.PUBLIC/name) for name in names}
    logs.append(execute(['scripts/report_m3w_european_cap_auxiliary_cost.py'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_cap_auxiliary_cost.py'], 'figure_replay'))
    logs.append(execute(['scripts/diagnose_m3w_european_cap_auxiliary_cost.py'], 'diagnosis_replay'))
    for name, digest in before.items(): assert run.digest(run.PUBLIC/name) == digest
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not suite.findall('.//failure') and not suite.findall('.//error') and not suite.findall('.//skipped')
    a = json.loads((run.PUBLIC/'training_replay.json').read_text())
    b = json.loads((run.PUBLIC/'readout_replay.json').read_text())
    assert a['heads'] == 432 and a['updates'] == 864000 and a['matched_arm_checks'] == 288
    assert b['groups'] == 36 and b['views'] == 144 and b['exact'] and b['direct_component_MSE_checks'] == 1152
    sources = set(run.FILES+TESTS+['scripts/plot_m3w_european_cap_auxiliary_cost.py',
                                 'scripts/diagnose_m3w_european_cap_auxiliary_cost.py',
                                 'scripts/verify_m3w_european_cap_auxiliary_cost.py'])
    doc = dict(all_passed=True, heads_replayed=432, readout_views_replayed=144,
        matched_arm_checks=288, direct_component_MSE_checks=1152, tests_passed=len(cases),
        scoped_test_files=len(TESTS), full_legacy_suite='not_run', report_and_figure_byte_reproducible=True,
        logs=logs, artifacts={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
                             if p.is_file() and p.name != 'verification.json'},
        source_bindings={p: run.digest(ROOT/p) for p in sorted(sources)})
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ('artifacts', 'source_bindings', 'logs')}, indent=2))


if __name__ == '__main__': main()
