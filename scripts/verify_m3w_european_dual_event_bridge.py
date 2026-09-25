"""Independently restore bridge heads and check source exclusion and reports."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_dual_event_bridge as run
from scripts import build_m3w_european_selection_data as adapter
from scripts.report_m3w_european_dual_event_bridge import summarize, ablations, gates
from scripts.report_m3w_european_floor_relative import dump
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, _, bank, pid, identity = run.registration()
    training = run.checked_training(identity)
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text())
    assert done['identity'] == identity and done['all_passed']
    data = adapter.load(run.parent, pid)
    future_rosters = set(data['sites'])
    inputs = adapter.load(run.parent, pid)
    assert not set(inputs) & {'target_eval', 'valid', 'future', 'endpoint'}
    restore_checks = []
    for g in training['groups']:
        name = g['group']; seed = int(name.split('_seed')[1].split('_')[0])
        x, env, _, _, _ = run.selection_pair(data, bank, g['producer'], seed, g['controller'])
        n = min(len(x), cfg['replay_rows'])
        assert set(g['producer_roster']).isdisjoint(g['controller_roster'])
        assert not future_rosters & (set(g['producer_roster']) | set(g['controller_roster']))
        for task in run.TASKS:
            directory = run.PRIVATE/'heads'/(name+'_'+task)
            r = json.loads((directory/'complete.json').read_text())
            pred, state = run.inc.prior.previous.restored(r)
            assert state['preprocess']['training_sites'] == sorted(g['controller_roster'])
            assert state['step'] == 2000 and r['fit']['unknown_rows_sampled'] == 0
            with np.load(directory/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['scores'][:n], pred(x[:n], env[:n]))
            restore_checks.append(dict(group=name, task=task, prefix=n, exact=True))
    first = json.loads((run.PRIVATE/'heads/fold0_seed17_controller1_utility/complete.json').read_text())
    pilot = json.loads((run.PRIVATE/'pilot.json').read_text())
    assert pilot['fit']['step'] == 100 and first['fit']['step'] == 2000 and first['fit']['new_updates'] == 1900
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    aggregate = summarize(rows); a = ablations(rows)
    assert aggregate == json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    assert a == json.loads((run.PUBLIC/'paired_ablations.json').read_text())
    seed = json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    assert gates(aggregate, seed, a) == json.loads((run.PUBLIC/'gates.json').read_text())
    diagnostic = json.loads((run.PUBLIC/'diagnostic_accounting.json').read_text())
    assert len(diagnostic['groups']) == 18 and not diagnostic['thresholds_changed']
    for r in diagnostic['groups'].values():
        assert len(r['by_locality']) == 6
        for v in r['by_locality'].values():
            np.testing.assert_allclose(v['dual_added_ADE_sum'], v['positive_gain']-v['positive_harm'], atol=1e-7)
    previous = json.loads((run.parent.PUBLIC/'verification.json').read_text())
    files = sorted(set(previous['test_files']) | {'tests/test_m3w_dual_event_bridge.py',
                                                 'tests/test_m3w_dual_event_bridge_reporting.py'})
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '--junitxml='+str(xml)],
                       cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr)
    print(p.stdout[-3500:], flush=True)
    if p.returncode: raise ValueError('Regression failures')
    root = ET.parse(xml).getroot(); suites = list(root) if root.tag == 'testsuites' else [root]
    assert not any(int(s.attrib.get('failures', 0))+int(s.attrib.get('errors', 0)) for s in suites)
    count = sum(int(s.attrib['tests']) for s in suites)
    required = ('conclusions.md', 'failure_analysis.md', 'model_card.md', 'data_card.md',
                'operation_zh.md', 'project_gap.md', 'results.md', 'world_model_gate.md')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*')
                 if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 1024**2 for f in artifacts)
    dump(run.PUBLIC/'verification.json', dict(all_passed=True, artifacts=artifacts,
        independent_checkpoint_prefix_replays=restore_checks, pilot_resumed_inside_budget=True,
        source_exclusion_checked_heads=54, matched_draws_and_statistics=training['matched_training_draws'],
        tests=count, test_files=files, full_legacy_suite='not_run', scope='67_focused_research_files',
        result_source='fresh_run_checks_cached_verified_frozen_assets',
        independent_calibration=False, independent_confirmation=False, deployment_changed=False,
        source_bindings={f: run.digest(ROOT/f) for f in [*run.FILES, *files,
            'scripts/report_m3w_european_dual_event_bridge.py', 'scripts/verify_m3w_european_dual_event_bridge.py',
            'scripts/diagnose_m3w_european_dual_event_bridge.py']}))
    print(json.dumps(dict(tests=count, files=len(files), replayed_heads=54, complete=True)))


if __name__ == '__main__': main()
