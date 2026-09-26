"""Replay every fitting-only diagnostic and seal scoped implementation evidence."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_fixed_cap_diagnostic as run


def main():
    run.registration()
    subprocess.run([sys.executable, 'scripts/run_m3w_european_fixed_cap_diagnostic.py', '--phase', 'verify'], cwd=ROOT, check=True)
    a = json.loads((run.PUBLIC/'completion.json').read_text())
    assert a == json.loads((run.PUBLIC/'replay.json').read_text())
    assert len(a['views']) == 144 and a['exact_identities'] == 2304
    names = ['summary.json', 'view_metrics.csv', 'risk_bin_metrics.csv', 'results.md']
    before = {name: run.digest(run.PUBLIC/name) for name in names}
    p = subprocess.run([sys.executable, 'scripts/run_m3w_european_fixed_cap_diagnostic.py', '--phase', 'report'], cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'report_replay.log').write_text(p.stdout+'\n'+p.stderr)
    assert p.returncode == 0, p.stderr
    assert all(run.digest(run.PUBLIC/name) == h for name, h in before.items())
    tests = ['tests/test_m3w_fixed_cap_diagnostic.py', 'tests/test_m3w_risk_conditioned_residual.py',
             'tests/test_m3w_event_transport.py', 'tests/test_m3w_context_residual.py',
             'tests/test_m3w_nested_residual.py', 'tests/test_m3w_nested_residual_reporting.py',
             'tests/test_m3w_nested_residual_accounting.py']
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable, '-m', 'pytest', *tests, '-q', '--junitxml='+str(xml)], cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr)
    print(p.stdout, flush=True); assert p.returncode == 0
    suites = list(ET.parse(xml).getroot())
    assert all(int(s.attrib.get('errors', 0))+int(s.attrib.get('failures', 0)) == 0 for s in suites)
    count = sum(int(s.attrib['tests']) for s in suites)
    h = run.digest(run.PUBLIC/'fixed_cap_diagnostic.svg')
    subprocess.run([sys.executable, 'scripts/plot_m3w_european_fixed_cap_diagnostic.py'], cwd=ROOT, check=True)
    assert run.digest(run.PUBLIC/'fixed_cap_diagnostic.svg') == h
    for f in ['conclusions.md', 'failure_analysis.md', 'method_note.md', 'operation_zh.md', 'project_gap.md', 'resource_observation.md']:
        assert (run.PUBLIC/f).is_file()
    artifacts = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 2**20 for f in artifacts)
    bindings = sorted(set(run.FILES+tests+['scripts/plot_m3w_european_fixed_cap_diagnostic.py', str(Path(__file__).relative_to(ROOT))]))
    doc = dict(all_passed=True, artifacts=artifacts, source_bindings={f: run.digest(ROOT/f) for f in bindings},
        fitting_views_replayed=144, exact_identities=2304, fresh_tests=count, test_files=tests,
        figure_byte_reproducible=True, architecture=platform.machine(), threads=4, workers=0,
        parent_replay_and_tests='cached_verified', full_legacy_suite='not_run',
        new_model_fit=False, held_labels_evaluated=False, conditional_bias_identified=False,
        independent_confirmation=False, deployment_changed=False)
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps(dict(tests=count, public_artifacts=len(artifacts), source_bindings=len(bindings), all_passed=True)))


if __name__ == '__main__':
    main()
