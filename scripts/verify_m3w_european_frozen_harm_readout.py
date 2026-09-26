"""Fresh readout replay and independent metric checks; no deployment certification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_frozen_harm_readout as run
from scripts.evaluate_m3w_european_support_fractional import measure
from scripts.report_m3w_european_frozen_harm_readout import contrasts,summarize_contrasts,gates_for,summarize_population,fit_transport
from src.evaluation.m3w_easy_membership_diagnosis import decompose,summary as membership_summary
from scripts.verify_m3w_european_harm_tail_crossfit import rank_metrics,top_share
import numpy as np
import torch


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'; binding=run.artifact(Path(__file__))
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['source_binding']==binding and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json'); return r
    records=[]; independent=0
    for g,data,pairs in run.parent.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            bx,env,by,*_=run.parent.base.pair_inputs(g,data,pairs,pair)
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            for held in sorted(set(sites)):
                tr,te,source_pr,cut,y,masks=run.parent.previous.fold_inputs(bx,env,by,cv,sites,held)
                target=run.parent.previous.diagnostic.event_targets(by[te],cv[te],cut)
                tag=name+'_'+pair+'_'+held; original,control=run.parent.base.restore(run.sources(tag)['mean_features'])
                train_ref=run.parent.base.method.predict(original,bx[tr],env[tr],source_pr)
                with np.load(run.sources(tag)['mean_features']/'scores.npz',allow_pickle=False) as z: ref=z['scores'].copy()
                fold=next(f for f in row['folds'] if f['held']==held)
                assert fold['target_sha256']==run.array_hash(target) and fold['held_ids_sha256']==run.array_hash(bi[te])
                initial=[]
                for arm in cfg['arms']:
                    encoder,source=run.parent.base.restore(run.sources(tag)[arm]); h=run.method.extract(encoder,bx,source_pr)
                    pr=run.method.preprocess(h[tr],source_pr); directory=run.PRIVATE/'heads'/tag/arm
                    model,state=run.method.restore(directory); r=json.loads((directory/'complete.json').read_text())
                    assert r['input']['train_latent_sha256']==run.array_hash(h[tr]) and r['input']['held_latent_sha256']==run.array_hash(h[te])
                    assert r['input']['train_y_sha256']==run.array_hash(y)
                    assert held not in source_pr['training_sites'] and held not in g['producer_roster']
                    assert state['step']==2000 and r['fit']['parameters']==130
                    for k in ('mean','std','known','weights'): np.testing.assert_array_equal(pr[k],state['preprocess'][k])
                    assert pr['cost_scale']==state['preprocess']['cost_scale']
                    for k in ('draws','fixed_ids'): np.testing.assert_array_equal(state[k],control[k])
                    assert torch.equal(state['sampler_rng'],control['sampler_rng']) and state['draws'][~pr['known']].sum()==0
                    initial.append(state['trace'][0])
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); pred=z['scores'].copy()
                    n=min(4096,te.sum()); np.testing.assert_array_equal(run.method.predict(model,h[te][:n],env[te][:n],pr,ref[:n]),pred[:n])
                    np.testing.assert_array_equal(pred[:,[0,2]],ref[:,[0,2]])
                    assert (pred[:,3]<=pred[:,1]).all() and (pred[:,1]<=env[te]+1e-5).all()
                    train_score=run.method.predict(model,h[tr],env[tr],pr,train_ref)
                    fit=json.loads((directory/'fit_diagnosis.json').read_text())
                    assert run.parent.previous.training_edges(train_score,env[tr],pr,cfg)==fit['edges']
                    assert run.parent.previous.diagnostic.summarize(train_score,y,env[tr],sites[tr],fit['edges'])==fit['training']
                    metrics=measure(pred,target,env[te],sites[te],fit['edges']); assert metrics==fold['metrics'][arm]
                    assert decompose(ref,pred,target,cv[te],env[te],cut)==fold['membership'][arm]
                    for subset,mask in (('all',np.ones(te.sum(),bool)),('envelope_positive',env[te]>0)):
                        known=np.isfinite(target).all(1)&mask
                        if not known.any(): continue
                        harm=target[known,3]
                        for key,score in run.parent.previous.diagnostic.score_columns(pred,env[te]).items():
                            au,ap=rank_metrics(score[known],harm); tail=top_share(score[known],harm,.1)
                            for actual,label in ((au,'AUROC'),(ap,'AUPRC'),(tail,'top10_harm_mass_share')):
                                expected=metrics[subset]['scores'][key][label]
                                if actual is None: assert expected is None
                                else: np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
                                independent+=1
                    records.append(dict(group=name,pair=pair,held=held,arm=arm,checkpoint=run.artifact(directory/'checkpoint.pt'),
                        prefix_rows=int(n),matched_sampler=True,reference_unchanged=True))
                    run.beat('verified_readout',completed=len(records),group=name,pair=pair,held=held,arm=arm)
                assert initial[0]==initial[1]
    assert len(records)==288
    r=dict(identity=identity,source_binding=binding,training=run.artifact(run.PRIVATE/'training_complete.json'),
        records=records,independent_rank_tail_checks=independent,all_passed=True)
    run.immutable_json(path,r); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--replay-only',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); done=run.checked_training(identity); receipt=replay(cfg,identity)
    if args.replay_only: print(json.dumps(dict(checkpoints=len(receipt['records']),checks=receipt['independent_rank_tail_checks']))); return
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    assert run.artifact(ROOT/checks['source_binding']['path'])==checks['source_binding']
    agg=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    assert contrasts(rows,cfg)==agg['contrasts'] and summarize_contrasts(agg['contrasts'])==agg['summary']
    assert gates_for(agg['summary'])==json.loads((run.PUBLIC/'gates.json').read_text())
    assert fit_transport(rows)==agg['fit_transport']
    fits=[json.loads((ROOT/ref['path']).read_text()) for ref in done['heads']]
    training=[dict(group=r['input']['group']['group'],pair=r['input']['pair'],held=r['input']['held'],
        arm=r['input']['arm'],fit=r['fit']) for r in fits]
    assert training==json.loads((run.PUBLIC/'training_metrics.json').read_text())
    compute=json.loads((run.PUBLIC/'compute_receipt.json').read_text())
    assert compute['heads']==len(fits)==288 and compute['updates']==sum(r['fit']['step'] for r in fits)==576000
    assert compute['summed_fit_seconds']==sum(r['fit']['seconds'] for r in fits)
    assert compute['unknown_draws']==sum(r['fit']['unknown_rows_sampled'] for r in fits)==0
    remote=compute['create_queue_observation']['receipt']
    assert run.artifact(ROOT/remote['path'])==remote
    remote_record=json.loads((ROOT/remote['path']).read_text())
    assert remote_record['response']['returncode']==0 and remote_record['jobs_submitted']==0 and not remote_record['remote_modified']
    for pair in cfg['pairs']:
        for subset in ('all','envelope_positive'):
            for arm in ('original_mean','original_fractional',*cfg['arms']):
                assert summarize_population([f['metrics'][arm][subset] for r in rows if r['pair']==pair for f in r['folds']])==agg['populations'][pair][subset][arm]
    for a in cfg['arms']:
        assert membership_summary([dict(r,folds=[dict(diagnosis=f['membership'][a]) for f in r['folds']]) for r in rows])==agg['membership'][a]
    old=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    tests=sorted(set(old['test_files'])|{'tests/test_m3w_frozen_harm_readout.py','tests/test_m3w_easy_membership_diagnosis.py','tests/test_m3w_frozen_harm_reporting.py'})
    xml=run.PRIVATE/'tests.xml'; p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2000:],flush=True); assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    assert all((run.PUBLIC/f).is_file() for f in ('conclusions.md','failure_analysis.md','model_card.md','data_card.md','operation_zh.md','project_gap.md','training_loss.svg','paired_contrasts.svg'))
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<1024**2 for f in artifacts)
    bindings=[*run.FILES,*tests,'scripts/diagnose_m3w_european_easy_membership.py','src/evaluation/m3w_easy_membership_diagnosis.py',
        'scripts/evaluate_m3w_european_frozen_harm_readout.py','scripts/report_m3w_european_frozen_harm_readout.py',
        'scripts/plot_m3w_european_frozen_harm_readout.py',str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,source_bindings={p:run.digest(ROOT/p) for p in bindings},
        tests=count,test_files=tests,full_legacy_suite='not_run',checkpoint_replays=len(receipt['records']),
        independent_rank_tail_checks=receipt['independent_rank_tail_checks'],policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(tests=count,files=len(tests),all_passed=True)))


if __name__=='__main__': main()
