"""Replay fitting, held costs and reports without reselecting any policy."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cost_mass as run
from scripts.verify_m3w_european_regime_transport import TESTS as PARENT_TESTS

TESTS = ['tests/test_m3w_cost_mass.py', 'tests/test_m3w_cost_mass_report.py',
    'tests/test_m3w_cost_mass_serialization.py', *PARENT_TESTS]


def execute(args, label):
    path = run.PRIVATE/(label+'.log')
    with path.open('w') as handle:
        result = subprocess.run([sys.executable,*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{label} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = []
    for phase in ('verify_fit','verify_eval'):
        logs.append(execute(['scripts/run_m3w_european_cost_mass.py','--phase',phase],phase))
    names = ['aggregate_metrics.json','fitting_diagnostics.json','fitting_diagnostics.md',
        'compute_receipt.json','results.md','mass_vs_raw.svg','mass_vs_scaled.svg',
        'auxiliary_mass.svg',*[f'mass_guard_{i}.svg' for i in range(3)]]
    before = {p:run.digest(run.PUBLIC/p) for p in names}
    logs.append(execute(['scripts/report_m3w_european_cost_mass_native.py'],'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_cost_mass.py'],'plot_replay'))
    assert all(run.digest(run.PUBLIC/p) == h for p,h in before.items())
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m','pytest','-q',*TESTS,'--junitxml',str(xml)],'tests'))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure','error','skipped'))
    for name in ('fitting_replay.json','evaluation_replay.json'):
        assert json.loads((run.PUBLIC/name).read_text())['exact']
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    assert len(rows) == 36 and sum(len(r['folds']) for r in rows) == 144
    checklist = run.PUBLIC/'reproducibility_checklist.md'
    pending = '- Final fitting, metric, report and figure replay: pending.'
    passed = (f'- Final replay completed:432 scalar fits,144 held views,2592 direct MSE checks;'
        f' reports and six figures byte-match;{len(cases)} tests in{len(TESTS)} scoped files pass.')
    content = checklist.read_text(); assert pending in content or passed in content
    checklist.write_text(content.replace(pending,passed))
    sources = sorted(set(run.FILES+TESTS+['scripts/verify_m3w_european_cost_mass.py',
        'scripts/plot_m3w_european_cost_mass.py','scripts/report_m3w_european_cost_mass_native.py']))
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,
        scalar_fits_replayed=432,source_held_views_replayed=144,direct_MSE_checks=2592,
        parent_raw_L2_metrics_exact=True,new_neural_heads=0,
        reports_byte_reproducible=True,scoped_tests_passed=len(cases),scoped_test_files=len(TESTS),
        full_legacy_suite='not_run',independent_confirmation=False,deployment_changed=False,
        logs=logs,local_detailed_metrics=[run.artifact(run.PUBLIC/'readout.json')],
        artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
            if p.is_file() and p.name not in ('verification.json','readout.json')},
        source_bindings={p:run.digest(ROOT/p) for p in sources}))
    print(json.dumps(dict(all_passed=True,scoped_tests_passed=len(cases),scoped_test_files=len(TESTS))))


if __name__ == '__main__': main()
