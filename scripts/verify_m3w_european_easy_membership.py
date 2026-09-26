"""Replay classification inputs/checkpoints and independently check probability errors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_easy_membership as run
from scripts.report_m3w_european_easy_membership import aggregates
from scripts.verify_m3w_european_harm_tail_crossfit import rank_metrics
import numpy as np
import torch


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'; binding=run.artifact(Path(__file__))
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['source_binding']==binding and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json'); return r
    records=[]; checks=0
    for g,data,pairs in run.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            x,env,by,*_=run.base.pair_inputs(g,data,pairs,pair)
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            for held in sorted(set(sites)):
                tr,te,pr,cut,oldy,_=run.tail.fold_inputs(x,env,by,cv,sites,held)
                y=run.method.labels(cv[tr],cut); target=run.method.labels(cv[te],cut)
                tag=name+'_'+pair+'_'+held; olddir=run.parent.sources(tag)['mean_features']; _,control=run.base.restore(olddir)
                fold=next(f for f in row['folds'] if f['held']==held); assert fold['target_sha256']==run.array_hash(target)
                assert fold['held_ids_sha256']==run.array_hash(bi[te]) and fold['easy_cut']==cut
                assert held not in pr['training_sites'] and held not in g['producer_roster']
                prevalence=float(pr['weights']@np.nan_to_num(y,nan=0.)); assert fold['train_prevalence']==prevalence
                with np.load(olddir/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); ref=z['scores'].copy()
                ratio=np.clip(np.divide(ref[:,2],ref[:,0],out=np.zeros(len(ref)),where=ref[:,0]>0),0,1)
                assert run.measure(ratio,target,env[te],sites[te])==fold['metrics']['reference_ratio']
                assert run.measure(np.full(te.sum(),prevalence),target,env[te],sites[te])==fold['metrics']['train_constant']
                initial=[]
                for arm in cfg['arms']:
                    directory=run.PRIVATE/'heads'/tag/arm; model,state=run.method.restore(directory)
                    r=json.loads((directory/'complete.json').read_text()); inp=r['input']
                    for key,value in (('train_x_sha256',x[tr]),('held_x_sha256',x[te]),('train_y_sha256',y),('train_ids_sha256',bi[tr]),('held_ids_sha256',bi[te])):
                        assert inp[key]==run.array_hash(value)
                    assert state['step']==2000 and state['prevalence']==prevalence
                    assert r['fit']['parameters']==(384 if arm=='linear' else 24641)
                    for k in ('mean','std','known','weights'): np.testing.assert_array_equal(pr[k],state['preprocess'][k])
                    for k in ('draws','fixed_ids'): np.testing.assert_array_equal(state[k],control[k])
                    assert torch.equal(state['sampler_rng'],control['sampler_rng']) and state['draws'][~pr['known']].sum()==0
                    initial.append(state['trace'][0])
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); pred=z['scores'].copy()
                    n=min(4096,te.sum()); np.testing.assert_array_equal(run.method.predict(model,x[te][:n],pr),pred[:n])
                    training=run.measure(run.method.predict(model,x[tr],pr),y,env[tr],sites[tr])
                    assert training==fold['training'][arm]['model']
                    assert run.measure(np.full(tr.sum(),prevalence),y,env[tr],sites[tr])==fold['training'][arm]['train_constant']
                    measured=run.measure(pred,target,env[te],sites[te]); assert measured==fold['metrics'][arm]
                    for subset,mask in (('all',np.ones(len(target),bool)),('envelope_positive',env[te]>0)):
                        known=np.isfinite(target)&mask
                        if not known.any(): continue
                        p,t=pred[known].astype(float),target[known]
                        np.testing.assert_allclose(float(((p-t)**2).mean()),measured[subset]['Brier'],rtol=1e-12,atol=1e-12); checks+=1
                        q=np.clip(p,1e-7,1-1e-7)
                        np.testing.assert_allclose(float(-(t*np.log(q)+(1-t)*np.log1p(-q)).mean()),measured[subset]['log_loss'],rtol=1e-12,atol=1e-12); checks+=1
                        au,ap=rank_metrics(p,t)
                        for v,k in ((au,'AUROC'),(ap,'AUPRC')):
                            if v is None: assert measured[subset][k] is None
                            else: np.testing.assert_allclose(v,measured[subset][k],rtol=1e-10,atol=1e-10)
                            checks+=1
                    records.append(dict(group=name,pair=pair,held=held,arm=arm,prefix_rows=int(n),checkpoint=run.artifact(directory/'checkpoint.pt')))
                    run.beat('verified',completed=len(records),group=name,pair=pair,held=held,arm=arm)
                assert initial[0]==initial[1]
    assert len(records)==288
    r=dict(identity=identity,source_binding=binding,training=run.artifact(run.PRIVATE/'training_complete.json'),records=records,
        independent_probability_rank_checks=checks,all_passed=True)
    run.immutable_json(path,r); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--replay-only',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); done=run.checked_training(identity); receipt=replay(cfg,identity)
    if args.replay_only: print(json.dumps(dict(checkpoints=len(receipt['records']),checks=receipt['independent_probability_rank_checks']))); return
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
    ref=comp['recent_CREATE_observation']['receipt']; assert run.artifact(ROOT/ref['path'])==ref
    old=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    tests=sorted(set(old['test_files'])|{'tests/test_m3w_easy_membership_probe.py','tests/test_m3w_easy_membership_metrics.py','tests/test_m3w_easy_membership_reporting.py'})
    xml=run.PRIVATE/'tests.xml'; p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2000:],flush=True); assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    required=('conclusions.md','failure_analysis.md','model_card.md','data_card.md','operation_zh.md','project_gap.md','training_loss.svg','paired_contrasts.svg')
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<2**20 for f in artifacts)
    bindings=[*run.FILES,*tests,'scripts/report_m3w_european_easy_membership.py','scripts/plot_m3w_european_easy_membership.py',str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,source_bindings={p:run.digest(ROOT/p) for p in bindings},
        tests=count,test_files=tests,full_legacy_suite='not_run',checkpoint_prefix_replays=len(receipt['records']),
        independent_probability_rank_checks=receipt['independent_probability_rank_checks'],policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(tests=count,files=len(tests),all_passed=True)))


if __name__=='__main__': main()
