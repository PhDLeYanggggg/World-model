"""Independent continuation replay, frozen-reference check and scoped regressions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_reference_protection as run
from scripts.report_m3w_european_reference_protection import summarize,matched_fit_summary
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from scripts.audit_m3w_easy_harm_fitting import summarize_fit
from src.world_model.m3w_native_forecast import draw_batch
import numpy as np
import torch


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'
    bindings={p:run.digest(ROOT/p) for p in [*run.FILES,str(Path(__file__).relative_to(ROOT))]}
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['bindings']==bindings and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json')
        return r
    records=[]; bfit=[]; decisions=0; mass_checks=0; protected_rows=0
    for g,data,pairs in run.parent.contexts(identity['parent']['parent']):
        name=g['group']; bi,ci=pairs['B']['ids'],pairs['C']['ids']
        queries=run.parent.query_groups(data['sites'][ci],data['recordings'][ci],data['frames'][ci])
        for pair in cfg['pairs']:
            bx,be,by,masks,pr,cx,ce,old,*_,meta=run.parent.pair_inputs(g,data,pairs,pair)
            input_ref=json.loads((run.PRIVATE/'inputs'/(name+'_'+pair+'.json')).read_text())
            assert meta==json.loads((ROOT/input_ref['parent_input']['path']).read_text())
            assert not set(meta['training_sites']) & set(meta['readout_sites'])
            warm=run.parent.PRIVATE/'heads'/(name+'_'+pair+'_mean'); initial,control=run.parent.restore(warm)
            assert run.artifact(warm/'checkpoint.pt')==input_ref['warm_start']
            rng=torch.Generator().set_state(control['sampler_rng']); draw_counts=np.zeros(len(bx),np.int64)
            groups=[np.flatnonzero(pr['known'] & (data['sites'][bi]==s)) for s in sorted(set(data['sites'][bi]))]
            for _ in range(cfg['head_training']['steps']):
                np.add.at(draw_counts,draw_batch(groups,cfg['head_training']['batch_size'],rng),1)
            with np.load(warm/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); warm_score=z['scores'].copy()
            fitting={'uniform':summarize_fit(run.parent.method.predict(initial,bx,be,pr),by,pr['weights'],
                pr['cost_scale'],control['loss_scales'],masks[:,2])}
            scores={}
            for arm in cfg['arms']:
                directory=run.PRIVATE/'heads'/(name+'_'+pair+'_'+arm); model,state=run.method.restore(directory)
                assert state['step']==2000
                for key in ('mean','std','known','weights'):
                    np.testing.assert_array_equal(pr[key],state['preprocess'][key])
                assert pr['cost_scale']==state['preprocess']['cost_scale']
                np.testing.assert_array_equal(state['loss_scales'],control['loss_scales'])
                np.testing.assert_array_equal(state['fixed_ids'],control['fixed_ids'])
                np.testing.assert_array_equal(draw_counts,state['draws'])
                assert torch.equal(rng.get_state(),state['sampler_rng'])
                assert state['draws'][~pr['known']].sum()==0
                for key,v in initial.state_dict().items(): assert torch.equal(v,state['initial_model'][key])
                if arm=='protected':
                    for key,v in model.reference.state_dict().items(): assert torch.equal(v,initial.state_dict()[key])
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ci); scores[arm]=z['scores'].copy()
                count=min(4096,len(ci))
                np.testing.assert_array_equal(run.parent.method.predict(model,cx[:count],ce[:count],pr),scores[arm][:count])
                if arm=='protected':
                    np.testing.assert_array_equal(scores[arm][:,[0,2]],warm_score[:,[0,2]]); protected_rows+=len(ci)
                fitting[arm]=summarize_fit(run.parent.method.predict(model,bx,be,pr),by,pr['weights'],
                    pr['cost_scale'],control['loss_scales'],masks[:,2])
                records.append(dict(group=name,pair=pair,arm=arm,prefix_rows=count,exact=True,
                    same_initial_weights=True,matched_continuation_draws=True,matched_final_sampler_rng=True))
            bfit.append(dict(group=name,pair=pair,arms=fitting))
            with np.load(run.prior.PRIVATE/'decisions'/(name+'_'+pair+'.npz'),allow_pickle=False) as z:
                previous={v:z[v].copy() for v in run.CONTROLS.values()}
            actions=run.actions(previous,scores,old['neural__utility'],old['moving'],ce,queries,ci)
            with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'.npz'),allow_pickle=False) as z:
                for key,v in actions.items(): np.testing.assert_array_equal(v,z[key]); decisions+=1
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            priorrow=json.loads((run.prior.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            for key,v in run.CONTROLS.items(): assert row['views'][key]==priorrow['views'][v]
            for arm in cfg['arms']:
                for site,value in row['query_budget'][arm].items():
                    other=row['views'][arm+'_joint']['risk']['by_locality'][site]
                    assert value['unknown_actions']==other['unknown_selected']
                    for event,e in value['events'].items():
                        np.testing.assert_allclose([e['actual_reference_mass'],e['actual_selected_harm']],
                            [other['events'][event]['reference_mass'],other['events'][event]['positive_harm']],rtol=1e-10,atol=1e-8)
                        mass_checks+=1
            run.beat('verified_reference_pair',group=name,pair=pair)
    result=dict(identity=identity,bindings=bindings,training=run.artifact(run.PRIVATE/'training_complete.json'),
        checkpoints=records,decisions=decisions,mass_checks=mass_checks,protected_C_row_views=protected_rows,
        full_B_fitting=bfit,result_source='fresh_run_frozen_model_replay_and_B_diagnosis',all_passed=True,changed_decisions=False)
    run.immutable_json(path,result); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--replay-only',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); run.checked_training(identity); r=replay(cfg,identity)
    if args.replay_only:
        print(json.dumps(dict(checkpoints=len(r['checkpoints']),decisions=r['decisions'],all_passed=True))); return
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['all_passed']
    rows={p:[] for p in cfg['pairs']}
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref
        row=json.loads((ROOT/ref['path']).read_text()); rows[row['pair']].append(row)
    seeds={p:seed_summary(v,cfg) for p,v in rows.items()}
    assert seeds==json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    assert {p:summarize(v,seeds[p]) for p,v in rows.items()}==json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    fits=json.loads((run.PUBLIC/'training_metrics.json').read_text())
    assert matched_fit_summary(fits)==json.loads((run.PUBLIC/'fitting_diagnostics.json').read_text())
    pv=json.loads((run.prior.PUBLIC/'verification.json').read_text())
    tests=sorted(set(pv['test_files'])|{'tests/test_m3w_reference_protection.py','tests/test_m3w_reference_protection_reporting.py'})
    xml=run.PRIVATE/'tests.xml'
    p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2400:],flush=True)
    if p.returncode: raise ValueError('Scoped regressions failed')
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert not any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    required=['results.md','conclusions.md','failure_analysis.md','model_card.md','data_card.md','operation_zh.md',
        'project_gap.md','world_model_gate.md','training_loss.svg','source_contrasts.svg']
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<1024**2 for f in artifacts)
    bindings=[*run.FILES,*tests,'scripts/report_m3w_european_reference_protection.py',
        'scripts/plot_m3w_european_reference_protection.py',str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={f:run.digest(ROOT/f) for f in bindings},checkpoint_replays=r['checkpoints'],
        decisions=r['decisions'],independent_event_mass_checks=r['mass_checks'],protected_C_row_views=r['protected_C_row_views'],
        tests=count,test_files=tests,full_legacy_suite='not_run',deployment_changed=False,independent_confirmation=False))
    print(json.dumps(dict(tests=count,test_files=len(tests),checkpoints=len(r['checkpoints']),decisions=r['decisions'],all_passed=True)))


if __name__=='__main__': main()
