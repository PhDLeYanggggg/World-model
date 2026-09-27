"""Replay nested lineage, magnitude fits, held costs and scoped tests."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_oof_magnitude as run
from scripts.verify_m3w_european_aux_prior import TESTS as PREVIOUS_TESTS
TESTS = ['tests/test_m3w_oof_magnitude.py', 'tests/test_m3w_oof_magnitude_training.py',
    'tests/test_m3w_oof_magnitude_report.py', 'tests/test_m3w_oof_checkpoint_recovery.py', *PREVIOUS_TESTS]


def execute(args, label):
    path = run.PRIVATE/(label+'.log')
    with path.open('w') as handle:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = []
    for phase in ('verify_training', 'verify_readouts', 'verify_eval'):
        logs.append(execute(['scripts/run_m3w_european_oof_magnitude.py', '--phase', phase], phase))
    names = ['aggregate_metrics.json', 'compute_receipt.json', 'results.md', 'source_held_costs.svg',
        'fitting_diagnostics.json', 'fitting_diagnostics.md']
    before = {p:run.digest(run.PUBLIC/p) for p in names}
    logs.append(execute(['scripts/report_m3w_european_oof_magnitude.py'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_oof_magnitude.py'], 'plot_replay'))
    logs.append(execute(['scripts/diagnose_m3w_european_oof_magnitude.py'], 'fitting_diagnostics_replay'))
    assert all(run.digest(run.PUBLIC/p) == h for p,h in before.items())
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    for name in ('training_replay.json', 'magnitude_replay.json', 'evaluation_replay.json'):
        assert json.loads((run.PUBLIC/name).read_text())['exact']
    prior = {r['group']+'_'+r['pair']:r for r in json.loads((run.previous.PUBLIC/'readout.json').read_text())['rows']}
    raw_controls = 0
    for row in json.loads((run.PUBLIC/'readout.json').read_text())['rows']:
        old = {f['held']:f for f in prior[row['group']+'_'+row['pair']]['folds']}
        for fold in row['folds']:
            for arm, previous in [('cost_only_raw', 'cost_only'), ('cap_aux_raw', 'new_true'), ('shuffled_aux_raw', 'new_shuffled')]:
                for subset in ('all', 'envelope_positive'):
                    assert fold['metrics'][arm][subset]['component_MSE'] == old[fold['held']]['metrics'][previous][subset]['component_MSE']
                    raw_controls += 1
    assert raw_controls == 864
    sources = sorted(set(run.FILES+TESTS+['scripts/verify_m3w_european_oof_magnitude.py',
        'scripts/plot_m3w_european_oof_magnitude.py', 'scripts/diagnose_m3w_european_oof_magnitude.py',
        'scripts/recover_m3w_oof_magnitude_checkpoint.py']))
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True,
        reference_lineages_replayed=144, inner_checkpoints_replayed=864,
        magnitude_fits_replayed=432, source_held_views_replayed=144,
        raw_control_component_MSE_exact_matches=raw_controls,
        reports_byte_reproducible=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS),
        full_legacy_suite='not_run', independent_confirmation=False, deployment_changed=False,
        logs=logs, local_detailed_metrics=[run.artifact(run.PUBLIC/'readout.json')],
        artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
            if p.is_file() and p.name not in ('verification.json', 'readout.json')},
        source_bindings={p:run.digest(ROOT/p) for p in sources}))
    print(json.dumps(dict(all_passed=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS))))


if __name__ == '__main__': main()
