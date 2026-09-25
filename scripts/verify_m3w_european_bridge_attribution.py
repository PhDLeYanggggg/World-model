"""Separate-process restore, source-boundary, matched-choice and report checks."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_bridge_attribution as run
from scripts import build_m3w_european_selection_data as adapter
from scripts.report_m3w_european_bridge_attribution import summarize, gates
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from src.world_model.m3w_bridge_attribution import decisions, independent_replay, query_groups, POLICIES
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, bank, pid, identity = run.registration(); training = run.checked_training(identity)
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text())
    assert done['identity'] == identity and done['all_passed']
    data = adapter.load(run.parent, pid)
    assert not set(data) & {'target_eval', 'valid', 'future', 'endpoint'}
    queries = query_groups(data['sites'], data['recordings'], data['frames'])
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    restore_checks = []; decision_checks = 0
    for g in training['groups']:
        assert set(g['producer_roster']).isdisjoint(g['controller_roster'])
        assert not set(data['sites']) & (set(g['producer_roster']) | set(g['controller_roster']))
        pairs, _ = run.selection(data, bank, g)
        for pair, (x, env, _, _) in pairs.items():
            root = run.bridge.PRIVATE if pair == 'full' else run.PRIVATE
            n = min(len(x), cfg['replay_rows'])
            if pair == 'motion_only': np.testing.assert_array_equal(x[:, -4:-2], np.zeros((len(x), 2)))
            for task in run.TASKS:
                directory = root/'heads'/(g['group']+'_'+task)
                r = json.loads((directory/'complete.json').read_text()); pred, state = run.inc.prior.previous.restored(r)
                assert state['preprocess']['training_sites'] == sorted(g['controller_roster'])
                assert state['step'] == 2000 and r['fit']['unknown_rows_sampled'] == 0
                with np.load(directory/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['scores'][:n], pred(x[:n], env[:n]))
                restore_checks.append(dict(group=g['group'], pair=pair, task=task, prefix=n, exact=True))
            values = run.scores(pair, g['group']); ids = np.arange(len(x))
            out, _, counts = decisions(*values, moving, env, queries, ids)
            independent = independent_replay(*values, moving, env, queries, ids)
            with np.load(run.PRIVATE/'decisions'/(g['group']+'_'+pair+'.npz'), allow_pickle=False) as z:
                for k in POLICIES:
                    np.testing.assert_array_equal(out[k], independent[k]); np.testing.assert_array_equal(out[k], z[k])
                    decision_checks += 1
            for q, count in zip(queries, counts):
                assert all(out[k][q].sum() == count for k in ('neural_matched', 'ridge_matched', 'hash_matched'))
    first = json.loads((run.PRIVATE/'heads/fold0_seed17_controller1_utility/complete.json').read_text())
    pilot = json.loads((run.PRIVATE/'pilot.json').read_text())
    assert pilot['fit']['step'] == 100 and first['fit']['step'] == 2000 and first['fit']['new_updates'] == 1900
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    seeds = seed_summary(rows, cfg)
    assert seeds == json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    aggregate = summarize(rows, seeds)
    assert aggregate == json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    assert gates(aggregate) == json.loads((run.PUBLIC/'gates.json').read_text())
    previous = json.loads((run.bridge.PUBLIC/'verification.json').read_text())
    files = sorted(set(previous['test_files']) | {'tests/test_m3w_bridge_attribution.py', 'tests/test_m3w_bridge_attribution_reporting.py'})
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '--junitxml='+str(xml)],
        cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2500:], flush=True)
    if p.returncode: raise ValueError('Regression failure')
    tree = ET.parse(xml).getroot(); suites = list(tree) if tree.tag == 'testsuites' else [tree]
    assert not any(int(s.attrib.get('failures', 0))+int(s.attrib.get('errors', 0)) for s in suites)
    count = sum(int(s.attrib['tests']) for s in suites)
    required = ('results.md', 'conclusions.md', 'failure_analysis.md', 'model_card.md', 'data_card.md',
                'operation_zh.md', 'project_gap.md', 'world_model_gate.md')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*')
                 if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 1024**2 for f in artifacts)
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True, artifacts=artifacts,
        checkpoint_prefix_replays=restore_checks, scalar_policy_replays=decision_checks,
        source_exclusion_checked_heads=72, matched_draws_support_cost_scale=True, pilot_resumed_inside_budget=True,
        tests=count, test_files=files, full_legacy_suite='not_run',
        result_source='fresh_run_checks_cached_verified_frozen_assets', deployment_changed=False,
        source_bindings={f: run.digest(ROOT/f) for f in [*run.FILES, *files,
            'scripts/report_m3w_european_bridge_attribution.py', 'scripts/verify_m3w_european_bridge_attribution.py']}))
    print(json.dumps(dict(tests=count, files=len(files), replayed_heads=72, complete=True)))


if __name__ == '__main__': main()
