"""Replay causal cost inference, labels and independent held-locality errors."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_membership_cost as run
from scripts.report_m3w_european_membership_cost import aggregates
from scripts.verify_m3w_european_harm_tail_crossfit import rank_metrics
import numpy as np
import torch


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'; binding=run.artifact(Path(__file__))
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['source_binding']==binding and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json'); return r
    assert 'probability' not in inspect.signature(run.method.fit).parameters
    assert 'easy' not in inspect.signature(run.method.predict).parameters
    records=[]; checks=0
    for g,data,pairs in run.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            x,env,by,*_=run.base.pair_inputs(g,data,pairs,pair)
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            oldrow=json.loads((run.parent.parent.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,_=run.tail.fold_inputs(x,env,by,cv,sites,held)
                easy=run.parent.method.labels(cv[tr],cut); target=run.tail.diagnostic.event_targets(by[te],cv[te],cut)
                tag=name+'_'+pair+'_'+held; tref,href,control,olddir=run.reference(tag,x,tr,te,pr,env,bi)
                memberdir=run.parent.PRIVATE/'heads'/tag/'mlp'; member,ms=run.parent.method.restore(memberdir)
                ptrain=run.parent.method.predict(member,x[tr],pr)
                with np.load(memberdir/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); pheld=z['scores'].copy()
                n=min(4096,te.sum()); np.testing.assert_array_equal(run.parent.method.predict(member,x[te][:n],pr),pheld[:n])
                fold=next(f for f in row['folds'] if f['held']==held)
                oldfold=next(f for f in oldrow['folds'] if f['held']==held)
                assert fold['target_sha256']==oldfold['target_sha256']==run.array_hash(target)
                assert fold['held_ids_sha256']==run.array_hash(bi[te]) and fold['easy_cut']==cut
                assert fold['metrics']['original_mean']==oldfold['metrics']['original_mean']
                assert held not in pr['training_sites'] and held not in g['producer_roster']
                predictions={'original_mean':href}
                for arm in cfg['arms']:
                    directory=run.PRIVATE/'heads'/tag/arm; model,s=run.method.restore(directory)
                    r=json.loads((directory/'complete.json').read_text()); inp=r['input']
                    for k,a in (('train_x_sha256',x[tr]),('held_x_sha256',x[te]),('train_y_sha256',y),
                                ('train_easy_sha256',easy),('train_ids_sha256',bi[tr]),('held_ids_sha256',bi[te])):
                        assert inp[k]==run.array_hash(a)
                    assert not inp['membership_probability_used_in_fitting'] and not inp['held_targets_used_for_fit']
                    assert inp['membership']==run.artifact(memberdir/'complete.json') and s['step']==2000
                    assert r['fit']['parameters']==24706
                    for k in ('mean','std','known','weights'): np.testing.assert_array_equal(s['preprocess'][k],pr[k])
                    for k in ('draws','fixed_ids'): np.testing.assert_array_equal(s[k],control[k])
                    assert torch.equal(s['sampler_rng'],control['sampler_rng']) and s['draws'][~pr['known']].sum()==0
                    assert s['prevalence']==ms['prevalence']
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); pred=z['scores'].copy()
                        if arm=='conditional': predictions['constant']=z['constant'].copy()
                    np.testing.assert_array_equal(run.method.predict(model,x[te][:n],env[te][:n],pr,arm=arm,probability=pheld[:n]),pred[:n][:,[1,3]])
                    predictions[arm]=pred
                    train=run.compose(run.method.predict(model,x[tr],env[tr],pr,arm=arm,probability=ptrain),tref)
                    edges=run.tail.training_edges(train,env[tr],pr,cfg)
                    diag=run.tail.diagnostic.summarize(train,y,env[tr],sites[tr],edges)
                    assert edges==fold['training'][arm]['edges'] and diag==fold['training'][arm]['training']
                    if arm=='conditional':
                        pc=np.full(te.sum(),ms['prevalence'])
                        np.testing.assert_array_equal(run.method.predict(model,x[te][:n],env[te][:n],pr,arm=arm,probability=pc[:n]),predictions['constant'][:n][:,[1,3]])
                        tc=run.compose(run.method.predict(model,x[tr],env[tr],pr,arm=arm,probability=np.full(tr.sum(),ms['prevalence'])),tref)
                        ce=run.tail.training_edges(tc,env[tr],pr,cfg)
                        assert ce==fold['training']['constant']['edges']
                        assert run.tail.diagnostic.summarize(tc,y,env[tr],sites[tr],ce)==fold['training']['constant']['training']
                    records.append(dict(group=name,pair=pair,held=held,arm=arm,prefix_rows=int(n),checkpoint=run.artifact(directory/'checkpoint.pt')))
                    run.beat('verified',completed=len(records),group=name,pair=pair,held=held,arm=arm)
                for arm,pred in predictions.items():
                    np.testing.assert_array_equal(pred[:,[0,2]],href[:,[0,2]])
                    assert run.measure(pred,target,env[te],sites[te],fold['training'][arm]['edges'])==fold['metrics'][arm]
                    if arm!='original_mean':
                        assert run.decompose(href,pred,target,cv[te],env[te],cut)==fold['membership'][arm]
                    for subset,mask in (('all',np.ones(te.sum(),bool)),('envelope_positive',env[te]>0)):
                        known=np.isfinite(target).all(1)&mask; m=fold['metrics'][arm][subset]
                        if not known.any(): continue
                        p,t=pred[known],target[known]
                        np.testing.assert_allclose(float(((p[:,3]-t[:,3])**2).mean()),m['harm_MSE'],rtol=1e-10,atol=1e-10); checks+=1
                        np.testing.assert_allclose(float(((p[:,1]-t[:,1])**2).mean()),m['component_MSE'][1],rtol=1e-12,atol=1e-12); checks+=1
                        au,ap=rank_metrics(p[:,3],t[:,3]>0)
                        for v,k in ((au,'AUROC'),(ap,'AUPRC')):
                            if v is None: assert m['scores']['moment'][k] is None
                            else: np.testing.assert_allclose(v,m['scores']['moment'][k],rtol=1e-10,atol=1e-10)
                            checks+=1
    assert len(records)==288
    r=dict(identity=identity,source_binding=binding,training=run.artifact(run.PRIVATE/'training_complete.json'),
        records=records,independent_cost_rank_checks=checks,all_passed=True)
    run.immutable_json(path,r); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--replay-only',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); done=run.checked_training(identity); receipt=replay(cfg,identity)
    if args.replay_only: print(json.dumps(dict(heads=len(receipt['records']),checks=receipt['independent_cost_rank_checks']))); return
    c=json.loads((run.PUBLIC/'completion_checks.json').read_text()); rows=[]
    for ref in c['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    agg=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text()); assert aggregates(rows,cfg)==agg
    assert agg['gates']==json.loads((run.PUBLIC/'gates.json').read_text())
    rs=[json.loads((ROOT/ref['path']).read_text()) for ref in done['heads']]
    fits=[dict(group=r['input']['group']['group'],pair=r['input']['pair'],held=r['input']['held'],arm=r['input']['arm'],fit=r['fit']) for r in rs]
    assert fits==json.loads((run.PUBLIC/'training_metrics.json').read_text())
    comp=json.loads((run.PUBLIC/'compute_receipt.json').read_text())
    assert comp['updates']==sum(r['fit']['step'] for r in fits)==576000 and comp['heads']==len(fits)==288
    assert comp['summed_fit_seconds']==sum(r['fit']['seconds'] for r in fits)
    assert comp['unknown_draws']==sum(r['fit']['unknown_rows_sampled'] for r in fits)==0
    remote=comp['create_queue']['receipt']; assert run.artifact(ROOT/remote['path'])==remote
    old=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    tests=sorted(set(old['test_files'])|{'tests/test_m3w_membership_cost.py','tests/test_m3w_membership_cost_reporting.py'})
    xml=run.PRIVATE/'tests.xml'; p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2000:],flush=True); assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    required=('conclusions.md','failure_analysis.md','project_gap.md','model_card.md','data_card.md','operation_zh.md','training_loss.svg','paired_contrasts.svg')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<2**20 for f in artifacts)
    bindings=[*run.FILES,*tests,'scripts/plot_m3w_european_membership_cost.py',str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,source_bindings={p:run.digest(ROOT/p) for p in bindings},
        tests=count,test_files=tests,full_legacy_suite='not_run',checkpoint_prefix_replays=len(receipt['records']),
        independent_cost_rank_checks=receipt['independent_cost_rank_checks'],policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(tests=count,files=len(tests),all_passed=True)))


if __name__=='__main__': main()
