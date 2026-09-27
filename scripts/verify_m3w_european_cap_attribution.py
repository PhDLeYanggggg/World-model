"""Single-use replay attestation; unchanged checks need only hash validation."""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_attribution as run
from scripts.verify_m3w_european_temporal_support import TESTS as PARENT_TESTS, execute

TESTS = ['tests/test_m3w_cap_attribution.py', 'tests/test_m3w_cap_attribution_report.py', *PARENT_TESTS]


def main():
    if (run.PUBLIC/'verification.json').exists(): raise ValueError('Immutable attestation exists; do not overwrite logs')
    directory = run.PRIVATE/'verification_runs'/str(time.time_ns()); directory.mkdir(parents=True, exist_ok=False)
    logs = []
    for phase in ('verify_predictions', 'verify_eval'):
        logs.append(execute(['scripts/run_m3w_european_cap_attribution.py', '--phase', phase], phase, directory))
    names = ['aggregate_metrics.json', 'gates.json', 'results.md', 'cap_attribution.svg']
    before = {p:run.digest(run.PUBLIC/p) for p in names}
    logs.append(execute(['scripts/report_m3w_european_cap_attribution.py'], 'report_replay', directory))
    logs.append(execute(['scripts/plot_m3w_european_cap_attribution.py'], 'plot_replay', directory))
    assert all(run.digest(run.PUBLIC/p) == h for p,h in before.items())
    xml = directory/'tests.xml'
    logs.append(execute(['-m', 'pytest', '-q', *TESTS, '--junitxml', str(xml)], 'tests', directory))
    suite = ET.parse(xml).getroot(); cases = suite.findall('.//testcase')
    assert cases and not any(suite.findall('.//'+k) for k in ('failure','error','skipped'))
    for name in ('prediction_replay.json','evaluation_replay.json'):
        assert json.loads((run.PUBLIC/name).read_text())['exact']
    sources = sorted(set(run.FILES+TESTS+['scripts/plot_m3w_european_cap_attribution.py',
        'scripts/verify_m3w_european_cap_attribution.py']))
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True, views_replayed=432,
        unprojected_vectors_replayed=1728, prior_scores_exact=True, new_fits=0, new_neural_heads=0,
        reports_and_figure_byte_reproducible=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS),
        full_legacy_suite='not_run', cold_rebuild_from_new_raw_downloads='not_run', independent_confirmation=False,
        logs=logs, local_detailed_metrics=[run.artifact(run.PRIVATE/'readout.json')],
        artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in sorted(run.PUBLIC.rglob('*'))
            if p.is_file() and p.name != 'verification.json'},
        source_bindings={p:run.digest(ROOT/p) for p in sources}))
    print(json.dumps(dict(all_passed=True, scoped_tests_passed=len(cases), scoped_test_files=len(TESTS))))


if __name__ == '__main__': main()
