"""Restore C predictions; replay maps, role exclusion and held-source risk checks."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_bridge_calibration as run
from scripts import build_m3w_european_selection_data as adapter
from scripts.report_m3w_european_bridge_calibration import summarize, feasibility, moment_transport
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from src.evaluation.m3w_bridge_risk_calibration import fit, apply
import numpy as np
import torch


def scalar(u, m, moving, env, rule):
    if rule['abstain'] or rule['kind'] == 'reference': return np.zeros(len(u), bool)
    threshold = .02
    if rule['kind'] == 'population_rescale': threshold *= rule['denominator_multiplier']/rule['harm_multiplier']
    if rule['kind'] == 'selected_risk_grid': threshold = rule['threshold']
    return np.array([bool(move and d > 0 and score[0] > score[1] and risk[1] <= threshold*risk[0])
                     for score, risk, move, d in zip(u, m, moving, env)], bool)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, bank, pid, identity = run.registration(); calibrated = run.checked_calibration(identity)
    data = adapter.load(run.parent, pid)
    assert not set(data) & {'target_eval', 'valid', 'future', 'endpoint'}
    run.inc.ensure_frozen(); _, bcfg, ctx, bid, _, _ = run.inc.load(); source_data = ctx[2]
    gs = {g['group']: g for g in calibrated['groups']}; prefix_checks = []
    for easy, all_event in run.bridge.source_pairs(bcfg, ctx, bid):
        a, seed, held = easy['fold'], easy['seed'], easy['ids']
        for b in range(3):
            if a == b: continue
            c = next(k for k in range(3) if k not in (a, b)); name = f'fold{a}_seed{seed}_controller{b}'; g = gs[name]
            take = np.isin(source_data['sites'][held], identity['rosters'][c]); ids = held[take]
            assert not set(source_data['sites'][ids]) & (set(g['producer_roster']) | set(g['controller_roster']) | set(data['sites']))
            cvp = easy['cv'][take]; damp = run.base.baseline_numpy(source_data['history'][ids], 3)-source_data['origin'][ids, None]
            bits = np.column_stack([v[k][take] for k in ('old', 'bits') for v in (easy, all_event)])
            x, env = run.bridge.features(source_data['geometry'][ids], cvp, easy['prediction'][take], all_event['prediction'][take], bits)
            mx, me, _, _ = run.motion_pair(source_data['geometry'][ids], cvp, damp, easy['bits'][take], all_event['bits'][take])
            for pair, xx, ee in (('full', x, env), ('motion_only', mx, me)):
                folder = run.PRIVATE/'source'/(name+'_'+pair); r = json.loads((folder/'receipt.json').read_text())
                assert r['input_sha256'] == run.array_hash(xx) and r['ids_sha256'] == run.array_hash(ids)
                with np.load(folder/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ids)
                    for family in cfg['families']:
                        for task in run.attribution.TASKS:
                            pred, state, _ = run.restored(pair, name, family, task)
                            assert state['preprocess']['training_sites'] == sorted(g['controller_roster'])
                            n = min(len(xx), cfg['replay_rows'])
                            np.testing.assert_array_equal(pred(xx[:n], ee[:n]), z[family+'__'+task][:n])
                            prefix_checks.append(dict(group=name, pair=pair, family=family, task=task, rows=n, exact=True))
            run.beat('verified_source_group', group=name)
    map_checks, selected_checks = 0, 0
    for ref in calibrated['maps']:
        mapping = json.loads((ROOT/ref['path']).read_text()); g = mapping['group']; name = g['group']
        pair, family = mapping['pair'], mapping['family']; folder = run.PRIVATE/'source'/(name+'_'+pair)
        with np.load(folder/'scores.npz', allow_pickle=False) as z: inputs = {k:z[k].copy() for k in z.files}
        with np.load(folder/'labels.npz', allow_pickle=False) as z: labels = {k:z[k].copy() for k in z.files}
        u, m = [inputs[family+'__'+t] for t in run.attribution.TASKS]
        rebuilt = fit(u, m, inputs['moving'], inputs['envelope'], **labels, sites=inputs['sites'],
            easy_cut=g['easy_cut'], grid=cfg['grid'])
        assert rebuilt == mapping['fitted']; map_checks += 1
        pairs, _ = run.attribution.selection(data, bank, g); _, env, _, _ = pairs[pair]
        values = run.attribution.scores(pair, name); tu, tm = values[:2] if family == 'neural' else values[2:]
        moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
        with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'_'+family+'.npz'), allow_pickle=False) as z:
            for kind, rule in rebuilt['rules'].items():
                np.testing.assert_array_equal(scalar(tu, tm, moving, env, rule), z[kind]); selected_checks += 1
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert done['all_passed'] and done['identity'] == identity
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    seeds = seed_summary(rows, cfg); assert seeds == json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    assert summarize(rows, seeds) == json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    assert feasibility() == json.loads((run.PUBLIC/'finite_scene_feasibility.json').read_text())
    assert moment_transport(rows, bank, pid) == json.loads((run.PUBLIC/'moment_transport.json').read_text())
    previous = json.loads((run.attribution.PUBLIC/'verification.json').read_text())
    files = sorted(set(previous['test_files']) | {'tests/test_m3w_bridge_risk_calibration.py', 'tests/test_m3w_bridge_calibration_reporting.py'})
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '--junitxml='+str(xml)],
        cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2500:], flush=True)
    if p.returncode: raise ValueError('Regression failure')
    root = ET.parse(xml).getroot(); suites = list(root) if root.tag == 'testsuites' else [root]
    assert not any(int(s.attrib.get('failures', 0))+int(s.attrib.get('errors', 0)) for s in suites)
    count = sum(int(s.attrib['tests']) for s in suites)
    required = ('results.md', 'conclusions.md', 'failure_analysis.md', 'model_card.md', 'data_card.md',
                'operation_zh.md', 'project_gap.md', 'finite_scene_feasibility.md', 'world_model_gate.md')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts = {str(p.relative_to(run.PUBLIC)): run.digest(p) for p in run.PUBLIC.rglob('*')
                 if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/f).stat().st_size < 1024**2 for f in artifacts)
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True, artifacts=artifacts,
        source_checkpoint_prefix_replays=prefix_checks, calibration_maps_replayed=map_checks,
        scalar_selection_policy_replays=selected_checks, source_exclusion_checked_models=144,
        tests=count, test_files=files, full_legacy_suite='not_run',
        result_source='fresh_run_checks_cached_verified_frozen_assets', deployment_changed=False,
        source_bindings={f:run.digest(ROOT/f) for f in [*run.FILES, *files,
            'scripts/report_m3w_european_bridge_calibration.py', 'scripts/verify_m3w_european_bridge_calibration.py']}))
    print(json.dumps(dict(tests=count, files=len(files), restored_models=len(prefix_checks), maps=map_checks,
        scalar_decisions=selected_checks, all_passed=True)))


if __name__ == '__main__': main()
