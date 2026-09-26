"""Replay fixed fits, frozen inference and readout; seal only inspected outputs."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_conditioned_residual as run


def main():
    run.registration()
    for phase in ('verify_support', 'verify_fit', 'verify_eval'):
        subprocess.run([sys.executable, 'scripts/run_m3w_european_risk_conditioned_residual.py', '--phase', phase], cwd=ROOT, check=True)
    before = {f: run.digest(run.PUBLIC/f) for f in ('aggregate_metrics.json', 'results.md')}
    subprocess.run([sys.executable, 'scripts/run_m3w_european_risk_conditioned_residual.py', '--phase', 'report'], cwd=ROOT, check=True)
    assert all(run.digest(run.PUBLIC/f) == h for f, h in before.items())
    support = json.loads((run.PUBLIC/'support_replay.json').read_text())
    assert support == json.loads((run.PUBLIC/'support_report.json').read_text())
    assert support['projection_checks'] == len(support['rows']) == 432
    check = json.loads((run.PUBLIC/'eval_replay.json').read_text())
    assert check['direct_MSE_checks'] == 1728 and len(check['groups']) == 36
    before_accounting = run.digest(run.PUBLIC/'error_accounting.json')
    subprocess.run([sys.executable, 'scripts/diagnose_m3w_european_risk_conditioned_residual.py'], cwd=ROOT, check=True)
    assert run.digest(run.PUBLIC/'error_accounting.json') == before_accounting
    tests = ['tests/test_m3w_risk_conditioned_residual.py', 'tests/test_m3w_event_transport.py',
             'tests/test_m3w_context_residual.py', 'tests/test_m3w_nested_residual.py',
             'tests/test_m3w_nested_residual_reporting.py', 'tests/test_m3w_nested_residual_accounting.py']
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable, '-m', 'pytest', *tests, '-q', '--junitxml='+str(xml)], cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout, flush=True); assert p.returncode == 0
    suites = list(ET.parse(xml).getroot()); count = sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('errors', 0))+int(s.attrib.get('failures', 0)) == 0 for s in suites)
    h = run.digest(run.PUBLIC/'risk_conditioned_residual.svg')
    subprocess.run([sys.executable, 'scripts/plot_m3w_european_risk_conditioned_residual.py'], cwd=ROOT, check=True)
    assert run.digest(run.PUBLIC/'risk_conditioned_residual.svg') == h
    for f in ('conclusions.md', 'failure_analysis.md', 'operation_zh.md', 'project_gap.md'):
        assert (run.PUBLIC/f).is_file()
    files = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 2**20 for f in files)
    bindings = sorted(set(run.FILES+tests+['scripts/plot_m3w_european_risk_conditioned_residual.py',
        'scripts/diagnose_m3w_european_risk_conditioned_residual.py',
        'scripts/diagnose_m3w_european_nested_residual.py', str(Path(__file__).relative_to(ROOT))]))
    doc = dict(all_passed=True, artifacts=files, source_bindings={f: run.digest(ROOT/f) for f in bindings},
        fitting_only_Torch_inference_replays=144, producer_projection_checks=432, probe_fit_prediction_replays=864,
        readout_groups_replayed=36, direct_MSE_checks=1728, fresh_tests=count, test_files=tests,
        post_freeze_error_identities=1728,
        parent_432_neural_head_replays='cached_verified', parent28_tests='cached_verified', full_legacy_suite='not_run',
        figure_byte_reproducible=True, architecture=platform.machine(), threads=4, workers=0,
        new_neural_updates=0, new_trajectory_updates=0, independent_confirmation=False, deployment_changed=False)
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps(dict(tests=count, public_artifacts=len(files), source_bindings=len(bindings), all_passed=True)))


if __name__ == '__main__': main()
