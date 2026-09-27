"""Seal completed raw replay and scoped tests; never infer scientific success."""
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_european_observation_quality as run

TESTS = ['tests/test_m3w_observation_quality.py', 'tests/test_m3w_masked_neighbor_integration.py',
         'tests/test_m3w_partial_context.py', 'tests/test_m3w_context_conditioning.py',
         'tests/test_m3w_european_source_forecast.py', 'tests/test_m3w_european_squares_source.py',
         'tests/test_m3w_temporal_support.py']


def main():
    seal = run.PUBLIC/'verification.json'
    if seal.exists(): raise ValueError('Existing seal: validate hashes rather than overwrite')
    run.registration()
    replay = json.loads((run.PUBLIC/'raw_replay.json').read_text())
    assert replay['exact'] and replay['audit_sha256'] == run.digest(run.PUBLIC/'audit.json')
    audit = json.loads((run.PUBLIC/'audit.json').read_text())
    for ref in audit['receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text())
        assert run.artifact(ROOT/r['diagnostics']['path']) == r['diagnostics']
    before = {n:run.digest(run.PUBLIC/n) for n in ('results.md', 'gates.json')}
    subprocess.run([sys.executable, 'scripts/report_m3w_european_observation_quality.py'], cwd=ROOT, check=True)
    assert all(run.digest(run.PUBLIC/n) == h for n,h in before.items())
    home = run.PRIVATE/'verification_runs'/str(time.time_ns()); home.mkdir(parents=True)
    xml = home/'tests.xml'; log = home/'tests.log'
    with log.open('w') as out:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)],
                       cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure', 'error', 'skipped'))
    source = sorted(set(run.FILES+TESTS+['scripts/report_m3w_european_observation_quality.py',
        'scripts/verify_m3w_european_observation_quality.py', 'src/world_model/m3w_partial_context.py',
        'src/world_model/m3w_european_source_forecast.py', 'src/world_model/m3w_native_forecast.py',
        'src/world_model/m3w_supervised_intervention.py', 'src/world_model/m3w_context_conditioning.py',
        'src/world_model/m3w_baseline_relative_forecaster.py', 'src/world_model/m3w_temporal_support.py',
        'src/data_unification/m3w_european_squares_source.py', 'src/evaluation/m3w_european_squares_raw_v2.py',
        'src/evaluation/m3w_european_squares_roles.py', 'src/evaluation/m3w_european_squares_intake.py']))
    run.immutable_json(seal, dict(raw_replay_exact=True, matched_observed_boxes=2551752,
        original_geometry_exact=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS),
        report_byte_reproducible=True, logs=[run.artifact(log), run.artifact(xml)],
        source_bindings={p:run.digest(ROOT/p) for p in source},
        artifacts={p.name:run.digest(p) for p in sorted(run.PUBLIC.iterdir()) if p.is_file()},
        full_legacy_suite='not_run', new_model_training=False, predictive_lift='not_run',
        independent_confirmation=False, deployment_changed=False))
    print(json.dumps(dict(raw_replay_exact=True, tests=len(cases), new_model_training=False)))


if __name__ == '__main__': main()
