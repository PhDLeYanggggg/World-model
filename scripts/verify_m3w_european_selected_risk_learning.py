"""Separate-process source reconstruction, checkpoint replay and report validation."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selected_risk_learning as run
from scripts.report_m3w_european_selected_risk_learning import summarize
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, identity = run.registration(); done = run.checked_training(identity)
    replays = []; decisions = 0
    for g, data, pairs in run.contexts(identity):
        ci = pairs['C']['ids']; name = g['group']
        queries = run.query_groups(data['sites'][ci], data['recordings'][ci], data['frames'][ci])
        for pair in cfg['pairs']:
            bx, be, by, masks, pr, cx, ce, old, bins, score, meta = run.pair_inputs(g, data, pairs, pair)
            recorded = json.loads((run.PRIVATE/'inputs'/(name+'_'+pair+'.json')).read_text())
            assert meta == recorded
            assert not set(meta['training_sites']) & set(meta['readout_sites'])
            new = {}
            root = run.previous.bridge.PRIVATE if pair == 'full' else run.previous.attribution.PRIVATE
            prior = torch.load(root/'heads'/(name+'_all_risk')/'checkpoint.pt', map_location='cpu', weights_only=False)
            for arm in cfg['arms']:
                directory = run.PRIVATE/'heads'/(name+'_'+pair+'_'+arm)
                model, s = run.restore(directory); assert s['step'] == 2000
                for k in ('mean', 'std', 'weights', 'known'): np.testing.assert_array_equal(pr[k], s['preprocess'][k])
                np.testing.assert_array_equal(s['draws'], prior['draws'])
                assert s['draws'][~pr['known']].sum() == 0 and s['preprocess']['cost_scale'] == pr['cost_scale']
                yy = np.where(pr['known'][:, None], by/pr['cost_scale'], 0).astype(np.float32)
                rms = np.sqrt(np.sum(pr['weights'][:, None]*yy.astype(float)**2, axis=0)).clip(1e-4)
                np.testing.assert_array_equal(rms, s['loss_scales'])
                with np.load(directory/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ci); new[arm] = z['scores'].copy()
                count = min(4096, len(cx))
                np.testing.assert_array_equal(run.method.predict(model, cx[:count], ce[:count], pr), new[arm][:count])
                replays.append(dict(group=name, pair=pair, arm=arm, rows=count, exact=True,
                    matched_original_draws=True, training_role='B_only'))
            actions = run.decision_bank(old, new, ce, queries, ci)
            with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'.npz'), allow_pickle=False) as z:
                for k,v in actions.items(): np.testing.assert_array_equal(v, z[k]); decisions += 1
                np.testing.assert_array_equal(z['support_bins'], bins); np.testing.assert_array_equal(z['support_score'], score)
            run.beat('verified_source_pair', group=name, pair=pair)
    result = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert result['all_passed']
    rows = {p:[] for p in cfg['pairs']}
    for ref in result['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text()); rows[r['pair']].append(r)
    seeds = {p:seed_summary(r, cfg) for p,r in rows.items()}
    assert seeds == json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    assert {p:summarize(r, seeds[p]) for p,r in rows.items()} == json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    parent = json.loads((run.previous.PUBLIC/'verification.json').read_text())
    files = sorted(set(parent['test_files']) | {'tests/test_m3w_selected_risk_learning.py', 'tests/test_m3w_selected_risk_reporting.py'})
    xml = run.PRIVATE/'tests.xml'
    p = subprocess.run([sys.executable, '-m', 'pytest', *files, '-q', '--junitxml='+str(xml)], cwd=ROOT,
                       capture_output=True, text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2300:], flush=True)
    if p.returncode: raise ValueError('Regression failure')
    suites = list(ET.parse(xml).getroot()); count = sum(int(s.attrib['tests']) for s in suites)
    assert not any(int(s.attrib.get('errors', 0))+int(s.attrib.get('failures', 0)) for s in suites)
    required = ['results.md', 'conclusions.md', 'failure_analysis.md', 'model_card.md', 'data_card.md',
                'operation_zh.md', 'project_gap.md', 'world_model_gate.md']
    assert all((run.PUBLIC/n).is_file() for n in required)
    artifacts = {str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name != 'verification.json'}
    assert all((run.PUBLIC/n).stat().st_size < 1024**2 for n in artifacts)
    bindings = [*run.FILES, *files, 'scripts/report_m3w_european_selected_risk_learning.py',
                'scripts/plot_m3w_european_selected_risk_learning.py',
                'scripts/verify_m3w_european_selected_risk_learning.py']
    run.immutable_json(run.PUBLIC/'verification.json', dict(all_passed=True, artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings}, checkpoint_replays=replays,
        scalar_and_vector_decisions=decisions, tests=count, test_files=files,
        full_legacy_suite='not_run', result_source='fresh_run_verification_cached_verified_frozen_models',
        independent_confirmation=False, deployment_changed=False))
    print(json.dumps(dict(tests=count, test_files=len(files), checkpoints=len(replays), decisions=decisions, all_passed=True)))


if __name__ == '__main__': main()
