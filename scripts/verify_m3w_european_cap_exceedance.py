"""Replay frozen predictions/readout and seal the exact verified evidence."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_exceedance as run

TESTS = ['tests/test_m3w_cap_exceedance.py', 'tests/test_m3w_cap_exceedance_reporting.py',
         'tests/test_m3w_fixed_cap_diagnostic.py', 'tests/test_m3w_risk_conditioned_residual.py',
         'tests/test_m3w_nested_residual.py', 'tests/test_m3w_context_residual.py']


def execute(args, name):
    path = run.PRIVATE/(name+'.log')
    with path.open('w') as handle:
        process = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'{name} failed; inspect {path}')
    return run.artifact(path)


def main():
    logs = []
    for phase in ['verify_training', 'verify_eval']:
        logs.append(execute(['scripts/run_m3w_european_cap_exceedance.py', '--phase', phase], phase))
    before = {name: run.digest(run.PUBLIC/name) for name in ['aggregate_metrics.json', 'results.md', 'cap_event_contrasts.svg']}
    logs.append(execute(['scripts/run_m3w_european_cap_exceedance.py', '--phase', 'report'], 'report_replay'))
    logs.append(execute(['scripts/plot_m3w_european_cap_exceedance.py'], 'figure_replay'))
    for name, h in before.items():
        assert run.digest(run.PUBLIC/name) == h
    xml = run.PRIVATE/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests'))
    suite = ET.parse(xml).getroot()
    cases = suite.findall('.//testcase')
    assert cases and not suite.findall('.//failure') and not suite.findall('.//error') and not suite.findall('.//skipped')
    checks = json.loads((run.PUBLIC/'training_replay.json').read_text())
    assert checks['heads'] == 288 and checks['updates'] == 576000
    reread = json.loads((run.PUBLIC/'readout_replay.json').read_text())
    assert reread['views'] == 144 and reread['exact']
    source_paths = set(run.FILES+TESTS+['scripts/plot_m3w_european_cap_exceedance.py',
                                      'scripts/verify_m3w_european_cap_exceedance.py'])
    doc = dict(all_passed=True, heads_replayed=288, readout_views_replayed=144,
               tests_passed=len(cases), scoped_test_files=len(TESTS), full_legacy_suite='not_run',
               report_and_figure_byte_reproducible=True, logs=logs,
               artifacts={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
                          if p.is_file() and p.name != 'verification.json'},
               source_bindings={p: run.digest(ROOT/p) for p in sorted(source_paths)})
    run.immutable_json(run.PUBLIC/'verification.json', doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ['artifacts', 'source_bindings', 'logs']}, indent=2))


if __name__ == '__main__': main()
