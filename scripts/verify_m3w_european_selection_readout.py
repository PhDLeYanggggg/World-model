"""Verify exclusion, frozen artifacts, all seed summaries and scoped regressions."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before Torch import')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selection_readout as run
from scripts.report_m3w_european_selection_readout import summarize
from scripts.report_m3w_european_floor_relative import dump
from scripts import build_m3w_european_selection_data as adapter


def main():
    cfg, bank, identity = run.registration()
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text())
    assert done['all_passed'] and done['identity'] == identity
    for ref in (done['summary'], done['data'], done['decisions']): assert run.artifact(ROOT/ref['path']) == ref
    packed = json.loads((run.PRIVATE/'packed/receipt.json').read_text())
    adapter.verify_arrays(run.PRIVATE/'packed', packed, run.digest)
    for ref in packed['records']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text())
        assert r['role'] == 'model_selection_reserved' and r['identity'] == identity
        adapter.verify_arrays((ROOT/ref['path']).parent, r, run.digest)
    references = {}
    for g in bank['groups'].values():
        references[g['forecaster']['path']] = g['forecaster']
        for ref in [g['floor_utility'], g['floor_risk'], *g['heads'].values()]:
            assert run.artifact(ROOT/ref['path']) == ref
            r = json.loads((ROOT/ref['path']).read_text())
            for a in r['artifacts'].values(): assert run.artifact(ROOT/a['path']) == a
            ck = r['artifacts'].get('checkpoint', r['artifacts'].get('model'))
            references[ck['path']] = ck
    fitting_roster = set(sum(bank['producer_rosters'], []))
    for path, ref in references.items():
        assert run.artifact(ROOT/path) == ref
        s = torch.load(ROOT/path, map_location='cpu', weights_only=False)
        sites = s['preprocess']['training_sites'] if 'preprocess' in s else s['identity']['fit_sites']
        assert sites and set(sites) <= fitting_roster and not set(sites) & set(cfg['localities'])
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text()); assert r['verified']; rows.append(r)
    assert summarize(rows) == json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    previous = json.loads((run.inc.PUBLIC/'completion_checks.json').read_text())
    files = sorted(set(previous['test_files']) | {'tests/test_m3w_frozen_selection.py',
        'tests/test_m3w_selection_readout_reporting.py', 'tests/test_m3w_european_squares_source.py',
        'tests/test_m3w_native_metrics.py'})
    xml = run.PRIVATE/'tests.xml'
    proc = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '--junitxml='+str(xml)],
        cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(proc.stdout+'\n'+proc.stderr)
    print(proc.stdout[-3500:], flush=True)
    if proc.returncode: raise ValueError('Regression suite failed')
    root = ET.parse(xml).getroot(); suites = list(root) if root.tag == 'testsuites' else [root]
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) == 0 for s in suites)
    count = sum(int(s.attrib['tests']) for s in suites)
    required = ('conclusions.md', 'failure_analysis.md', 'project_gap.md', 'model_card.md',
        'data_card.md', 'operation_zh.md', 'results.md')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*')
        if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 1024**2 for f in artifacts)
    bindings = {f: run.digest(ROOT/f) for f in [*run.FILES, *files,
        'scripts/report_m3w_european_selection_readout.py', 'scripts/verify_m3w_european_selection_readout.py',
        'scripts/diagnose_m3w_european_selection_readout.py']}
    dump(run.PUBLIC/'verification.json', dict(all_passed=True, artifacts=artifacts, bindings=bindings,
        source_excluded_checkpoints=len(references), selection_groups_verified=len(rows),
        record_receipts_verified=len(packed['records']), no_fitting_on_selection=True,
        tests=count, test_files=files, scoped_files=len(files), full_legacy_suite='not_run',
        scientific_success_not_implied=True, calibrated_safety=False, deployment_changed=False))
    print(json.dumps(dict(tests=count, scoped_files=len(files), checkpoints=len(references), public_artifacts=len(artifacts))))


if __name__ == '__main__': main()
