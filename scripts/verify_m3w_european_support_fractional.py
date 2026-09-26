"""Replay matched fits, held predictions and paired diagnostics before publication."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_support_fractional as run
from scripts.evaluate_m3w_european_support_fractional import measure
from scripts.report_m3w_european_support_fractional import paired_contrasts,summarize_population
from scripts.diagnose_m3w_european_support_fractional import fit_transport_summary
from src.evaluation.m3w_harm_error_decomposition import error_decomposition
from scripts.verify_m3w_european_harm_tail_crossfit import rank_metrics,top_share
import numpy as np
import torch


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'; source=run.artifact(Path(__file__))
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['source_binding']==source and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json'); return r
    records=[]; independent=0; coordinate=0
    for g,data,pairs in run.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            bx,env,by,*_=run.base.pair_inputs(g,data,pairs,pair)
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            errors=[]
            for pred in pairs['B'][pair][2:]:
                a=run.base.base.cross.independent.coordinate_errors(pred.astype(float)+data['origin'][bi,None],
                    data['target_eval'][bi],data['valid'][bi])[0]
                errors.append(a); coordinate+=1
            np.testing.assert_allclose(by[:,:2],np.column_stack((errors[0],np.maximum(errors[1]-errors[0],0))),
                rtol=1e-10,atol=1e-8,equal_nan=True)
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,masks=run.previous.fold_inputs(bx,env,by,data['baseline_ade'][bi,1],sites,held)
                tag=name+'_'+pair+'_'+held; directory=run.PRIVATE/'heads'/tag
                model,state=run.base.restore(directory); _,control=run.base.restore(run.previous.PRIVATE/'heads'/tag)
                assert state['step']==control['step']==2000 and state['coefficient']==1
                assert held not in pr['training_sites'] and held not in g['producer_roster']
                for key in ('draws','fixed_ids','loss_scales'): np.testing.assert_array_equal(state[key],control[key])
                assert torch.equal(state['sampler_rng'],control['sampler_rng'])
                for key in ('mean','std','known','weights'):
                    np.testing.assert_array_equal(state['preprocess'][key],pr[key])
                    np.testing.assert_array_equal(state['preprocess'][key],control['preprocess'][key])
                assert state['preprocess']['cost_scale']==pr['cost_scale']
                assert state['draws'][~pr['known']].sum()==0
                assert state['trace'][0]['moment_mse']==control['trace'][0]['moment_mse']
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); score=z['scores'].copy()
                n=min(4096,te.sum()); np.testing.assert_array_equal(run.base.method.predict(model,bx[te][:n],env[te][:n],pr),score[:n])
                target=run.previous.diagnostic.event_targets(by[te],data['baseline_ade'][bi[te],1],cut)
                assert (score[:,1]<=env[te]+1e-5).all() and (score[:,3]<=score[:,1]+1e-5).all()
                fit=json.loads((directory/'fit_diagnosis.json').read_text())
                train_score=run.base.method.predict(model,bx[tr],env[tr],pr)
                assert run.previous.training_edges(train_score,env[tr],pr,cfg)==fit['edges']
                assert run.previous.diagnostic.summarize(train_score,y,env[tr],sites[tr],fit['edges'])==fit['training']
                f=next(f for f in row['folds'] if f['held']==held)
                assert f['target_sha256']==run.array_hash(target) and f['held_ids_sha256']==run.array_hash(bi[te])
                arm_predictions={}
                for arm,home in (('mean',run.previous.PRIVATE),('fractional',run.PRIVATE)):
                    d=home/'heads'/tag
                    with np.load(d/'scores.npz',allow_pickle=False) as z: pred=z['scores'].copy()
                    arm_predictions[arm]=pred
                    edges=json.loads((d/'fit_diagnosis.json').read_text())['edges']
                    metrics=measure(pred,target,env[te],sites[te],edges)
                    assert metrics==f['metrics'][arm]
                    for subset,mask in (('all',np.ones(te.sum(),bool)),('envelope_positive',env[te]>0)):
                        known=np.isfinite(target).all(1)&mask
                        if not known.any(): continue
                        harm=target[known,3]
                        for key,values in run.previous.diagnostic.score_columns(pred,env[te]).items():
                            au,ap=rank_metrics(values[known],harm); tail=top_share(values[known],harm,.1)
                            for actual,label in ((au,'AUROC'),(ap,'AUPRC'),(tail,'top10_harm_mass_share')):
                                wanted=metrics[subset]['scores'][key][label]
                                if actual is None: assert wanted is None
                                else: np.testing.assert_allclose(actual,wanted,rtol=1e-10,atol=1e-10)
                                independent+=1
                breakdown=error_decomposition(arm_predictions['mean'],arm_predictions['fractional'],
                    target,env[te],fit['edges']['envelope'][-1])
                for arm in ('mean','fractional'):
                    np.testing.assert_allclose(breakdown[arm+'_MSE'],f['metrics'][arm]['envelope_positive']['harm_MSE'],
                        rtol=1e-10,atol=1e-10)
                records.append(dict(group=name,pair=pair,held=held,checkpoint=run.artifact(directory/'checkpoint.pt'),
                    prefix_rows=int(n),matching_sampler=True,training_bins_replayed=True,error_decomposition=breakdown))
                run.beat('verified_head',group=name,pair=pair,held=held,completed=len(records))
    assert len(records)==144
    r=dict(identity=identity,source_binding=source,training=run.artifact(run.PRIVATE/'training_complete.json'),
        records=records,independent_rank_tail_checks=independent,coordinate_reductions=coordinate,all_passed=True)
    run.immutable_json(path,r); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--replay-only',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); run.checked_training(identity); r=replay(cfg,identity)
    if args.replay_only: print(json.dumps(dict(checkpoints=len(r['records']),checks=r['independent_rank_tail_checks']))); return
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    assert run.artifact(ROOT/checks['source_binding']['path'])==checks['source_binding']
    aggregate=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    diagnosis=json.loads((run.PUBLIC/'fit_transport_diagnosis.json').read_text())
    assert diagnosis['summary']==fit_transport_summary(rows)
    assert run.artifact(ROOT/diagnosis['source_binding']['path'])==diagnosis['source_binding']
    assert paired_contrasts(rows,cfg)==aggregate['contrasts']
    for pair in cfg['pairs']:
        group=[r for r in rows if r['pair']==pair]
        for subset in ('all','envelope_positive'):
            for arm in ('mean','fractional'):
                assert summarize_population([f['metrics'][arm][subset] for row in group for f in row['folds']])==aggregate['populations'][pair][subset][arm]
        for key,summary in aggregate['summary'][pair].items():
            vals=[v[key] for v in aggregate['contrasts'][pair].values()]; good=[v for v in vals if 'CI' in v]
            assert summary==dict(positive=sum(v['CI'][0]>0 for v in good),negative=sum(v['CI'][1]<0 for v in good),
                overlap=sum(v['CI'][0]<=0<=v['CI'][1] for v in good),not_estimable=len(vals)-len(good),
                point_range=[min(v['point'] for v in good),max(v['point'] for v in good)] if good else None)
    gates=json.loads((run.PUBLIC/'gates.json').read_text()); full=aggregate['summary']['full']
    primary=full['envelope_positive__harm_MSE_gain_percent']['positive']==6
    guards=all(full['envelope_positive__'+key]['negative']==full['envelope_positive__'+key]['not_estimable']==0
        for key in ('top10_gain_pp','coverage_log_error_reduction'))
    assert gates['development_advance_gate']==(primary and guards)
    assert gates['primary_six_positive']==primary and gates['tail_and_coverage_guards']==guards
    old=json.loads((run.previous.PUBLIC/'verification.json').read_text())
    tests=sorted(set(old['test_files'])|{'tests/test_m3w_support_fractional_harm.py','tests/test_m3w_support_fractional_reporting.py'})
    xml=run.PRIVATE/'tests.xml'
    p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2500:],flush=True); assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    required=['results.md','conclusions.md','failure_analysis.md','model_card.md','data_card.md','operation_zh.md',
        'project_gap.md','training_loss.svg','paired_contrasts.svg']
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<1024**2 for f in artifacts)
    bindings=[*run.FILES,*tests,'scripts/evaluate_m3w_european_support_fractional.py',
        'scripts/report_m3w_european_support_fractional.py','scripts/plot_m3w_european_support_fractional.py',
        'scripts/diagnose_m3w_european_support_fractional.py',
        'src/evaluation/m3w_harm_error_decomposition.py',str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},tests=count,test_files=tests,full_legacy_suite='not_run',
        checkpoint_replays=len(r['records']),rank_tail_checks=r['independent_rank_tail_checks'],
        coordinate_reductions=r['coordinate_reductions'],policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(tests=count,files=len(tests),all_passed=True)))


if __name__=='__main__': main()
