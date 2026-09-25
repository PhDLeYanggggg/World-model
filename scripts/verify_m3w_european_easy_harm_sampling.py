"""Separate-process sampling, checkpoint, decision and reporting replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_easy_harm_sampling as run
from scripts.report_m3w_european_easy_harm_sampling import summarize, diagnostics
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from scripts.audit_m3w_easy_harm_fitting import summarize_fit
import numpy as np
import torch


def replay_draws(q, state, seed, settings):
    rng = torch.Generator().manual_seed(seed+7919)
    cdf = torch.tensor(q.cumsum(), dtype=torch.float64); cdf[-1] = 1.
    draws = np.zeros(len(q), dtype=np.int64)
    for _ in range(settings['steps']):
        ids = run.method.draw(cdf, settings['batch_size'], rng)
        np.add.at(draws, ids, 1)
    np.testing.assert_array_equal(draws, state['draws'])
    assert torch.equal(rng.get_state(), state['sampler_rng'])
    return dict(exact_draw_histogram=True, exact_final_rng=True, draws=int(draws.sum()))


def replay(cfg, identity):
    path = run.PUBLIC/'replay_receipt.json'
    binding = {p:run.digest(ROOT/p) for p in [*run.FILES,
        'scripts/verify_m3w_european_easy_harm_sampling.py','scripts/audit_m3w_easy_harm_fitting.py']}
    if path.exists():
        result = json.loads(path.read_text())
        assert result['identity']==identity and result['bindings']==binding and result['all_passed']
        assert result['training'] == run.artifact(run.PRIVATE/'training_complete.json')
        return result
    replays = []; fitting = []; decisions = 0; mass_checks = 0
    for g,data,pairs in run.parent.contexts(identity['parent']):
        name = g['group']; seed = int(name.split('_seed')[1].split('_')[0])
        bi,ci = pairs['B']['ids'],pairs['C']['ids']
        queries = run.parent.query_groups(data['sites'][ci],data['recordings'][ci],data['frames'][ci])
        for pair in cfg['pairs']:
            bx,be,by,masks,pr,cx,ce,old,*_,meta = run.parent.pair_inputs(g,data,pairs,pair)
            receipt = json.loads((run.PRIVATE/'inputs'/(name+'_'+pair+'.json')).read_text())
            assert run.artifact(ROOT/receipt['parent_inputs']['path']) == receipt['parent_inputs']
            assert meta == json.loads((ROOT/receipt['parent_inputs']['path']).read_text())
            assert not set(meta['training_sites']) & set(meta['readout_sites'])
            model,state = run.parent.restore(run.PRIVATE/'heads'/(name+'_'+pair))
            assert state['step'] == 2000
            old_model,control = run.parent.restore(run.parent.PRIVATE/'heads'/(name+'_'+pair+'_mean'))
            for k in ('mean','std','known','weights'):
                np.testing.assert_array_equal(pr[k], state['preprocess'][k])
            assert pr['cost_scale'] == state['preprocess']['cost_scale']
            q,w = run.method.probabilities(by,data['sites'][bi],pr['weights'])
            np.testing.assert_array_equal(q,state['probabilities'])
            np.testing.assert_array_equal(w,state['importance_ratio'])
            yy = np.where(pr['known'][:,None],by/pr['cost_scale'],0).astype(np.float32)
            rms = np.sqrt(np.sum(pr['weights'][:,None]*yy.astype(float)**2,axis=0)).clip(1e-4)
            np.testing.assert_array_equal(rms,state['loss_scales'])
            np.testing.assert_array_equal(rms,control['loss_scales'])
            assert state['trace'][0]['moment_mse'] == control['trace'][0]['moment_mse']
            assert run.expectation_check(by,data['sites'][bi],pr,control) == receipt['expectation']
            fitting.append(dict(group=name,pair=pair,result_source='fresh_run_full_B_inference',
                corrected=summarize_fit(run.parent.method.predict(model,bx,be,pr),by,pr['weights'],
                    pr['cost_scale'],rms,masks[:,2]),
                uniform=summarize_fit(run.parent.method.predict(old_model,bx,be,pr),by,pr['weights'],
                    pr['cost_scale'],rms,masks[:,2])))
            draws = replay_draws(q,state,seed,cfg['head_training'])
            assert state['draws'][~pr['known']].sum() == 0
            with np.load(run.PRIVATE/'heads'/(name+'_'+pair)/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); score = z['scores'].copy()
            n = min(4096,len(ci))
            np.testing.assert_array_equal(run.parent.method.predict(model,cx[:n],ce[:n],pr),score[:n])
            with np.load(run.parent.PRIVATE/'decisions'/(name+'_'+pair+'.npz'),allow_pickle=False) as z:
                prior = {k:z[k].copy() for k in run.CONTROL_KEYS}
            actions = run.actions(prior,score,old['neural__utility'],old['moving'],ce,queries,ci)
            with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'.npz'),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci)
                for k,v in actions.items(): np.testing.assert_array_equal(v,z[k]); decisions += 1
            row = json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            parent_row = json.loads((run.parent.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            for key in run.CONTROL_KEYS: assert row['views'][key] == parent_row['views'][key]
            for site,value in row['query_budget'].items():
                independent = row['views']['corrected_joint']['risk']['by_locality'][site]
                assert value['unknown_actions'] == independent['unknown_selected']
                for event,e in value['events'].items():
                    other = independent['events'][event]
                    np.testing.assert_allclose([e['actual_reference_mass'],e['actual_selected_harm']],
                        [other['reference_mass'],other['positive_harm']],rtol=1e-10,atol=1e-8)
                    mass_checks += 1
            replays.append(dict(group=name,pair=pair,prefix_rows=n,exact=True,sampling=draws,
                expected_loss_identity=True,training_role='B_only'))
            run.beat('verified_sampling_pair',group=name,pair=pair)
    result = dict(identity=identity,bindings=binding,training=run.artifact(run.PRIVATE/'training_complete.json'),
        checkpoints=replays,decisions=decisions,mass_checks=mass_checks,B_fitting=fitting,
        all_passed=True,result_source='fresh_run_replay_and_B_fitting_audit',decisions_changed=False)
    run.immutable_json(path,result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--replay-only',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, identity = run.registration(); run.checked_training(identity)
    replayed = replay(cfg,identity)
    if args.replay_only:
        print(json.dumps(dict(checkpoints=len(replayed['checkpoints']),decisions=replayed['decisions'],all_passed=True))); return
    checked = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checked['all_passed']
    rows = {p:[] for p in cfg['pairs']}
    for ref in checked['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text()); rows[r['pair']].append(r)
    seeds = {p:seed_summary(v,cfg) for p,v in rows.items()}
    assert seeds == json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    assert {p:summarize(v,seeds[p]) for p,v in rows.items()} == json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    training = json.loads((run.PUBLIC/'training_metrics.json').read_text())
    assert diagnostics(training,rows) == json.loads((run.PUBLIC/'diagnostic_summary.json').read_text())
    parent_verification = json.loads((run.parent.PUBLIC/'verification.json').read_text())
    tests = sorted(set(parent_verification['test_files']) | {'tests/test_m3w_easy_harm_sampling.py',
        'tests/test_m3w_easy_harm_reporting.py'})
    xml = run.PRIVATE/'tests.xml'
    result = subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],
        cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(result.stdout+'\n'+result.stderr); print(result.stdout[-2300:],flush=True)
    if result.returncode: raise ValueError('Scoped regression failure')
    suites = list(ET.parse(xml).getroot()); count = sum(int(s.attrib['tests']) for s in suites)
    assert not any(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0)) for s in suites)
    required = ['conclusions.md','failure_analysis.md','results.md','model_card.md','data_card.md',
        'operation_zh.md','project_gap.md','world_model_gate.md','training_loss.svg','source_contrasts.svg']
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts = {str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*')
        if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<1024**2 for f in artifacts)
    bindings = [*run.FILES,*tests,'scripts/report_m3w_european_easy_harm_sampling.py',
        'scripts/plot_m3w_european_easy_harm_sampling.py','scripts/audit_m3w_easy_harm_fitting.py',
        str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},checkpoint_replays=replayed['checkpoints'],
        decision_views=replayed['decisions'],independent_event_mass_checks=replayed['mass_checks'],
        tests=count,test_files=tests,full_legacy_suite='not_run',
        result_source='fresh_run_verification_cached_verified_frozen_models',
        independent_confirmation=False,deployment_changed=False))
    print(json.dumps(dict(checkpoints=len(replayed['checkpoints']),decisions=replayed['decisions'],mass_checks=replayed['mass_checks'],
        tests=count,test_files=len(tests),all_passed=True)))


if __name__ == '__main__': main()
